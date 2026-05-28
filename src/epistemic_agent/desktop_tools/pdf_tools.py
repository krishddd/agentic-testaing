"""
PDF Operations Tools

Create, read, and manipulate PDF documents.
Uses fpdf2 for creation, pypdf for reading.
Graceful fallback if libraries not installed.
"""

import os
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Graceful imports
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    logger.warning("[PDFTools] fpdf2 not installed — PDF creation disabled. Install: pip install fpdf2")

try:
    from pypdf import PdfReader, PdfWriter, PdfMerger
    HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter, PdfMerger
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False
        logger.warning("[PDFTools] pypdf not installed — PDF reading disabled. Install: pip install pypdf")


@dataclass
class PDFResult:
    """Result of a PDF operation."""
    success: bool
    operation: str
    message: str
    path: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None


class PDFTools:
    """
    PDF creation and manipulation tools.
    
    Capabilities:
    - Create PDFs from text/structured content
    - Read/extract text from existing PDFs
    - Merge multiple PDFs
    - Get PDF metadata
    """

    def create_pdf(
        self,
        content: str,
        output_path: str,
        title: str = "Document",
        author: str = "Epistemic Agent",
        font_size: int = 12,
    ) -> PDFResult:
        """
        Create a PDF from text content.
        Supports basic markdown-like formatting:
          # Heading 1
          ## Heading 2
          - Bullet points
          Regular paragraphs
        """
        if not HAS_FPDF:
            return PDFResult(
                success=False, operation="create_pdf", message="",
                error="fpdf2 not installed. Install: pip install fpdf2"
            )

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Set metadata
            pdf.set_title(title)
            pdf.set_author(author)
            pdf.set_creation_date(datetime.now())

            # Title
            pdf.set_font("Helvetica", "B", 18)
            pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)

            # Date
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(8)

            # Separator line
            pdf.set_draw_color(100, 100, 100)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)

            # Process content line by line
            for line in content.split("\n"):
                line = line.strip()
                
                if not line:
                    pdf.ln(4)
                    continue

                if line.startswith("# "):
                    # Heading 1
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.ln(4)
                    pdf.cell(0, 10, line[2:], new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                elif line.startswith("## "):
                    # Heading 2
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.ln(3)
                    pdf.cell(0, 9, line[3:], new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                elif line.startswith("### "):
                    # Heading 3
                    pdf.set_font("Helvetica", "BI", 12)
                    pdf.ln(2)
                    pdf.cell(0, 8, line[4:], new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                elif line.startswith("- ") or line.startswith("* "):
                    # Bullet point
                    pdf.set_font("Helvetica", "", font_size)
                    pdf.cell(8, 6, chr(8226))  # bullet char
                    pdf.multi_cell(0, 6, line[2:])
                elif line.startswith("**") and line.endswith("**"):
                    # Bold
                    pdf.set_font("Helvetica", "B", font_size)
                    pdf.multi_cell(0, 6, line.strip("*"))
                else:
                    # Regular paragraph
                    pdf.set_font("Helvetica", "", font_size)
                    pdf.multi_cell(0, 6, line)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            pdf.output(output_path)

            file_size = os.path.getsize(output_path)
            return PDFResult(
                success=True, operation="create_pdf",
                message=f"PDF created: {output_path} ({file_size} bytes, {pdf.page} pages)",
                path=output_path,
                data={"pages": pdf.page, "size_bytes": file_size, "title": title}
            )
        except Exception as e:
            return PDFResult(
                success=False, operation="create_pdf", message="",
                error=f"PDF creation failed: {str(e)}"
            )

    def read_pdf(self, filepath: str) -> PDFResult:
        """
        Extract text from an existing PDF file.
        """
        if not HAS_PYPDF:
            return PDFResult(
                success=False, operation="read_pdf", message="",
                error="pypdf not installed. Install: pip install pypdf"
            )

        if not os.path.exists(filepath):
            return PDFResult(
                success=False, operation="read_pdf", message="",
                error=f"File not found: {filepath}"
            )

        try:
            reader = PdfReader(filepath)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")

            full_text = "\n".join(text_parts)

            return PDFResult(
                success=True, operation="read_pdf",
                message=f"Read {len(reader.pages)} pages from {os.path.basename(filepath)}",
                path=filepath,
                data={
                    "pages": len(reader.pages),
                    "text": full_text[:5000],
                    "total_chars": len(full_text),
                    "metadata": dict(reader.metadata) if reader.metadata else {},
                }
            )
        except Exception as e:
            return PDFResult(
                success=False, operation="read_pdf", message="",
                error=f"PDF read failed: {str(e)}"
            )

    def merge_pdfs(self, file_list: List[str], output_path: str) -> PDFResult:
        """
        Merge multiple PDF files into one.
        """
        if not HAS_PYPDF:
            return PDFResult(
                success=False, operation="merge_pdfs", message="",
                error="pypdf not installed"
            )

        missing = [f for f in file_list if not os.path.exists(f)]
        if missing:
            return PDFResult(
                success=False, operation="merge_pdfs", message="",
                error=f"Files not found: {missing}"
            )

        try:
            merger = PdfMerger()
            for filepath in file_list:
                merger.append(filepath)

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            merger.write(output_path)
            merger.close()

            return PDFResult(
                success=True, operation="merge_pdfs",
                message=f"Merged {len(file_list)} PDFs into {output_path}",
                path=output_path,
                data={"input_files": len(file_list), "output": output_path}
            )
        except Exception as e:
            return PDFResult(
                success=False, operation="merge_pdfs", message="",
                error=f"Merge failed: {str(e)}"
            )

    def pdf_info(self, filepath: str) -> PDFResult:
        """
        Get metadata and info about a PDF file.
        """
        if not HAS_PYPDF:
            return PDFResult(
                success=False, operation="pdf_info", message="",
                error="pypdf not installed"
            )

        if not os.path.exists(filepath):
            return PDFResult(
                success=False, operation="pdf_info", message="",
                error=f"File not found: {filepath}"
            )

        try:
            reader = PdfReader(filepath)
            file_size = os.path.getsize(filepath)
            
            info = {
                "pages": len(reader.pages),
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 1),
                "encrypted": reader.is_encrypted,
            }
            
            if reader.metadata:
                info["metadata"] = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                    "creation_date": str(reader.metadata.get("/CreationDate", "")),
                }

            return PDFResult(
                success=True, operation="pdf_info",
                message=f"PDF: {len(reader.pages)} pages, {info['file_size_kb']} KB",
                path=filepath,
                data=info
            )
        except Exception as e:
            return PDFResult(
                success=False, operation="pdf_info", message="",
                error=f"Info extraction failed: {str(e)}"
            )
