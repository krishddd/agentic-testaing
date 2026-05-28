from truepy import License
import os
import datetime
import time
import ntplib
import logging
from functools import wraps # For creating robust retry decorator

# Configure basic logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Exception Classes ---
class LicenseError(Exception):
    """Base exception for license-related errors."""
    pass

class LicenseFileError(LicenseError):
    """Exception raised for errors in loading license or certificate files."""
    pass

class LicenseVerificationError(LicenseError):
    """Exception raised when license verification against certificate fails."""
    pass

class LicenseExpiredError(LicenseError):
    """Exception raised when the license has expired or is not yet valid."""
    pass

class NTPError(LicenseError):
    """Exception raised for issues with NTP server communication."""
    pass

# --- Helper for Retries ---
def retry(times: int, delay: float):
    """
    Decorator for retrying a function a specified number of times.
    Args:
        times: Number of times to retry.
        delay: Delay in seconds between retries.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Attempt {i+1}/{times} of '{func.__name__}' failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            raise # Re-raise the last exception if all retries fail
        return wrapper
    return decorator

# --- Updated get_internet_time function ---
@retry(times=3, delay=5) # Retry 3 times with 5 seconds delay
def get_internet_time(ntp_server: str = 'pool.ntp.org', timeout: float = 5.0) -> datetime.datetime:
    """
    Fetches the current time from an NTP server.
    Args:
        ntp_server: The NTP server to query. Defaults to 'pool.ntp.org'.
        timeout: The timeout for the NTP request in seconds.
    Returns:
        A datetime object representing the current internet time.
    Raises:
        NTPError: If there's any issue communicating with the NTP server.
    """
    try:
        client = ntplib.NTPClient()
        # Use version=3 for NTP, and specify timeout
        response = client.request(ntp_server, version=3, timeout=timeout)
        return datetime.datetime.fromtimestamp(response.tx_time)
    except ntplib.NTPException as e:
        raise NTPError(f"NTP server communication error: {e}")
    except OSError as e:
        # This catches network-related OS errors like host unreachable, no route to host
        raise NTPError(f"Network error during NTP request: {e}. Check internet connection.")
    except Exception as e:
        # Catch any other unexpected errors during NTP request
        raise NTPError(f"An unexpected error occurred during NTP request: {e}")

# --- Main check_license function ---
def check_license(
    license_dir: str = './License', # Default license directory
    certificate_filename: str = 'certificate.pem',
    license_filename: str = 'license.key',
    secret_key: bytes = b'SP@tai_2024' # Consider loading this from env var
) -> bool:
    """
    Performs a comprehensive check of the application license.

    Returns:
        True if the license is valid.
    Raises:
        LicenseFileError: If license or certificate files are missing or unreadable.
        LicenseVerificationError: If the license cannot be verified against the certificate.
        NTPError: If there's an issue fetching time from an NTP server.
        LicenseExpiredError: If the license is expired or not yet within its valid period.
    """
    try:
        cert_path = os.path.join(license_dir, certificate_filename)
        license_path = os.path.join(license_dir, license_filename)

        # 1. Load Certificate and License
        with open(cert_path, 'rb') as f:
            certificate = f.read()
        logging.info(f"Loaded certificate from {cert_path}")

        with open(license_path, 'rb') as f:
            # The secret_key should ideally come from a secure environment variable
            license_obj = License.load(f, secret_key)
        logging.info(f"Loaded license from {license_path}")

        # 2. Verify License Signature with Certificate
        license_obj.verify(certificate)
        logging.info("License signature verified against certificate successfully.")

    except FileNotFoundError as e:
        raise LicenseFileError(f"License file not found: {e.filename}. Ensure '{license_dir}' exists and contains '{certificate_filename}' and '{license_filename}'.")
    except Exception as e: # Catches errors from truepy's License.load or License.verify
        # This can catch the 'AttributeError: ... no attribute 'verifier'' if it's still present
        raise LicenseVerificationError(f"License integrity or verification failed: {e}. Check truepy and cryptography library versions and license file integrity.")

    # 3. Check License Time Validity using Internet Time
    try:
        current_time = get_internet_time()
        logging.info(f"Current UTC time from NTP: {current_time}")
    except NTPError as e:
        logging.error(f"Failed to get internet time for license check: {e}")
        raise # Re-raise the NTPError to indicate critical dependency failure

    start_period = license_obj.data.not_before
    expiry_period = license_obj.data.not_after

    logging.info(f"License valid from: {start_period}")
    logging.info(f"License valid to: {expiry_period}")

    if start_period <= current_time < expiry_period:
        logging.info("Valid License !!! Application is authorized to run.")
        return True
    else:
        raise LicenseExpiredError(
            f"License is expired or not yet valid. "
            f"Current time: {current_time}. Valid period: {start_period} to {expiry_period}."
        )
