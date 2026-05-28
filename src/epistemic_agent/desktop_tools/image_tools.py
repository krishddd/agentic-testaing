"""
Image Editing Tools

Create and edit images using Pillow.
Capabilities: create, resize, crop, convert, watermark, info.
Graceful fallback if Pillow not installed.
"""

import os
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Graceful import
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    logger.warning("[ImageTools] Pillow not installed — image tools disabled. Install: pip install Pillow")


@dataclass
class ImageResult:
    """Result of an image operation."""
    success: bool
    operation: str
    message: str
    path: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None


class ImageTools:
    """
    Image creation and editing for the epistemic agent.
    
    Capabilities:
    - Create images with text overlay
    - Resize, crop images
    - Convert between formats
    - Add watermarks
    - Get image metadata
    """

    def create_image(
        self,
        width: int = 800,
        height: int = 600,
        color: str = "white",
        text: str = "",
        output_path: str = "output.png",
        text_color: str = "black",
        font_size: int = 24,
    ) -> ImageResult:
        """
        Create a new image, optionally with text overlay.
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="create_image", message="",
                error="Pillow not installed. Install: pip install Pillow"
            )

        try:
            img = Image.new("RGB", (width, height), color)
            
            if text:
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                # Center text
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = (width - text_w) // 2
                y = (height - text_h) // 2
                draw.text((x, y), text, fill=text_color, font=font)

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            img.save(output_path)

            return ImageResult(
                success=True, operation="create_image",
                message=f"Image created: {output_path} ({width}x{height})",
                path=output_path,
                data={"width": width, "height": height, "format": output_path.split(".")[-1].upper()}
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="create_image", message="",
                error=f"Image creation failed: {str(e)}"
            )

    def resize_image(
        self,
        filepath: str,
        width: int,
        height: int,
        output_path: Optional[str] = None,
    ) -> ImageResult:
        """
        Resize an image to specified dimensions.
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="resize_image", message="",
                error="Pillow not installed"
            )

        if not os.path.exists(filepath):
            return ImageResult(
                success=False, operation="resize_image", message="",
                error=f"File not found: {filepath}"
            )

        try:
            img = Image.open(filepath)
            original_size = img.size
            resized = img.resize((width, height), Image.Resampling.LANCZOS)

            out = output_path or filepath
            os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
            resized.save(out)

            return ImageResult(
                success=True, operation="resize_image",
                message=f"Resized {original_size[0]}x{original_size[1]} -> {width}x{height}",
                path=out,
                data={"original": list(original_size), "new": [width, height]}
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="resize_image", message="",
                error=f"Resize failed: {str(e)}"
            )

    def crop_image(
        self,
        filepath: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        output_path: Optional[str] = None,
    ) -> ImageResult:
        """
        Crop an image to specified region.
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="crop_image", message="",
                error="Pillow not installed"
            )

        if not os.path.exists(filepath):
            return ImageResult(
                success=False, operation="crop_image", message="",
                error=f"File not found: {filepath}"
            )

        try:
            img = Image.open(filepath)
            cropped = img.crop((left, top, right, bottom))

            out = output_path or filepath
            os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
            cropped.save(out)

            return ImageResult(
                success=True, operation="crop_image",
                message=f"Cropped to ({left},{top})-({right},{bottom}) = {right-left}x{bottom-top}",
                path=out,
                data={"crop_box": [left, top, right, bottom], "size": [right-left, bottom-top]}
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="crop_image", message="",
                error=f"Crop failed: {str(e)}"
            )

    def convert_format(
        self,
        filepath: str,
        output_format: str = "PNG",
        output_path: Optional[str] = None,
    ) -> ImageResult:
        """
        Convert image between formats (PNG, JPEG, BMP, TIFF, WEBP).
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="convert_format", message="",
                error="Pillow not installed"
            )

        if not os.path.exists(filepath):
            return ImageResult(
                success=False, operation="convert_format", message="",
                error=f"File not found: {filepath}"
            )

        try:
            img = Image.open(filepath)
            original_format = img.format or "unknown"

            # Generate output path if not provided
            if not output_path:
                base = os.path.splitext(filepath)[0]
                ext = output_format.lower()
                if ext == "jpeg":
                    ext = "jpg"
                output_path = f"{base}.{ext}"

            # Convert RGBA to RGB for JPEG
            if output_format.upper() in ("JPEG", "JPG") and img.mode == "RGBA":
                img = img.convert("RGB")

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            img.save(output_path, format=output_format.upper())

            return ImageResult(
                success=True, operation="convert_format",
                message=f"Converted {original_format} -> {output_format.upper()}",
                path=output_path,
                data={"original_format": original_format, "new_format": output_format.upper()}
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="convert_format", message="",
                error=f"Conversion failed: {str(e)}"
            )

    def add_watermark(
        self,
        filepath: str,
        watermark_text: str,
        output_path: Optional[str] = None,
        opacity: int = 80,
        font_size: int = 36,
    ) -> ImageResult:
        """
        Add a text watermark to an image.
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="add_watermark", message="",
                error="Pillow not installed"
            )

        if not os.path.exists(filepath):
            return ImageResult(
                success=False, operation="add_watermark", message="",
                error=f"File not found: {filepath}"
            )

        try:
            img = Image.open(filepath).convert("RGBA")
            
            # Create watermark layer
            watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

            # Center watermark
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (img.width - text_w) // 2
            y = (img.height - text_h) // 2
            
            draw.text((x, y), watermark_text, fill=(128, 128, 128, opacity), font=font)

            # Composite
            result = Image.alpha_composite(img, watermark)
            result = result.convert("RGB")

            out = output_path or filepath
            os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
            result.save(out)

            return ImageResult(
                success=True, operation="add_watermark",
                message=f"Watermark '{watermark_text}' added to {os.path.basename(filepath)}",
                path=out,
                data={"watermark": watermark_text, "opacity": opacity}
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="add_watermark", message="",
                error=f"Watermark failed: {str(e)}"
            )

    def get_image_info(self, filepath: str) -> ImageResult:
        """
        Get metadata about an image.
        """
        if not HAS_PILLOW:
            return ImageResult(
                success=False, operation="get_image_info", message="",
                error="Pillow not installed"
            )

        if not os.path.exists(filepath):
            return ImageResult(
                success=False, operation="get_image_info", message="",
                error=f"File not found: {filepath}"
            )

        try:
            img = Image.open(filepath)
            file_size = os.path.getsize(filepath)

            info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 1),
            }

            return ImageResult(
                success=True, operation="get_image_info",
                message=f"Image: {img.width}x{img.height} {img.format} ({info['file_size_kb']} KB)",
                path=filepath,
                data=info
            )
        except Exception as e:
            return ImageResult(
                success=False, operation="get_image_info", message="",
                error=f"Info extraction failed: {str(e)}"
            )
