"""Render PDF pages into a complete, atomically published preparation bundle."""

import hashlib
import shutil
import tempfile
from importlib.metadata import version
from math import ceil
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from multimodal_doc_parser.document_schema.document import (
    AssetReference,
    Document,
    Page,
    PreparationSettings,
)


class DocumentPreparationError(ValueError):
    """An input cannot be prepared within the configured limits."""


def prepare_document(
    source_path: Path,
    output_root: Path,
    *,
    dpi: int = 144,
    max_pages: int = 200,
    max_page_pixels: int = 25_000_000,
) -> Path:
    """Return a new bundle directory; never overwrite existing results.

    PDFium calls must be serialized within a process. Use worker processes,
    rather than concurrent threads, when this function is integrated into a service.
    """
    if not 36 <= dpi <= 600:
        raise DocumentPreparationError("DPI must be between 36 and 600")
    if max_pages < 1 or max_page_pixels < 1:
        raise DocumentPreparationError("Page and pixel limits must be positive")
    if not source_path.is_file():
        raise DocumentPreparationError(f"Input file does not exist: {source_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=output_root))
    try:
        # Snapshot once so the content hash and rendered bytes always agree.
        snapshot = staging / "source.pdf"
        shutil.copyfile(source_path, snapshot)
        with snapshot.open("rb") as source_file:
            digest = hashlib.file_digest(source_file, "sha256").hexdigest()
        destination = output_root / f"{digest}-dpi-{dpi}"
        if destination.exists():
            raise DocumentPreparationError(f"Output already exists: {destination}")
        try:
            pdf = pdfium.PdfDocument(snapshot)
        except pdfium.PdfiumError as exc:
            raise DocumentPreparationError(
                "Cannot open PDF; it may be invalid or encrypted"
            ) from exc
        try:
            if len(pdf) == 0 or len(pdf) > max_pages:
                raise DocumentPreparationError(f"PDF must contain between 1 and {max_pages} pages")
            page_directory = staging / "assets" / "pages"
            page_directory.mkdir(parents=True)
            pages: list[Page] = []
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    width, height = page.get_size()
                    scale = dpi / 72
                    # Ceiling dimensions match the renderer; check before allocation.
                    if ceil(width * scale) * ceil(height * scale) > max_page_pixels:
                        raise DocumentPreparationError(f"Page {index + 1} exceeds the pixel limit")
                    relative_image = f"assets/pages/page_{index + 1:04d}.png"
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil()
                        try:
                            image.save(staging / relative_image, format="PNG")
                            pages.append(
                                Page(
                                    page_number=index + 1,
                                    width_points=width,
                                    height_points=height,
                                    width_pixels=image.width,
                                    height_pixels=image.height,
                                    rotation_degrees=page.get_rotation(),
                                    image=AssetReference(path=relative_image),
                                )
                            )
                        finally:
                            image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            document = Document(
                document_id=f"sha256:{digest}",
                source_filename=source_path.name,
                title=pdf.get_metadata_value("Title") or None,
                preparation=PreparationSettings(
                    dpi=dpi, renderer=f"pypdfium2:{version('pypdfium2')}"
                ),
                pages=pages,
            )
        finally:
            pdf.close()
        (staging / "document.json").write_text(
            document.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        # Existing complete bundles are nonempty, so rename cannot replace them.
        staging.rename(destination)
        return destination
    except pdfium.PdfiumError as exc:
        raise DocumentPreparationError("PDF page rendering failed") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
