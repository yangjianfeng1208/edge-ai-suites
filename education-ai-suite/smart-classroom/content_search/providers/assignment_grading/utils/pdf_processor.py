from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path


def convert_pdf_to_images(
    pdf_path: Path,
    dpi: int = 50,
    grayscale: bool = True,
    save_dir: Optional[Path] = None,
    jpeg_quality: int = 85,
    max_pixels: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    resize_ratio: Optional[float] = None,
    enhance_contrast: Optional[float] = None,
    enhance_sharpness: Optional[float] = None
) -> List[Image.Image]:
    """
    Convert PDF to images with optimization options.

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for rendering (50=fast/low-res, 150=medium, 300=high-res)
        grayscale: Convert to grayscale
        save_dir: Optional directory to save images
        jpeg_quality: JPEG quality (0-100), lower=faster but worse quality
        max_pixels: Max total pixels (e.g., 5000000 for 5MP), resize if exceeded
        max_width: Max width in pixels, resize if exceeded
        max_height: Max height in pixels, resize if exceeded
        resize_ratio: Direct resize ratio (e.g., 0.5 for 50% size)
        enhance_contrast: Contrast enhancement factor (1.0=no change, >1=more contrast)
        enhance_sharpness: Sharpness enhancement factor (1.0=no change, >1=sharper)

    Returns:
        List of PIL Image objects

    Speed optimization tips:
        - Lower DPI (30-50): Faster rendering, smaller images
        - Lower jpeg_quality (60-75): Faster encoding
        - Set max_pixels (e.g., 2000000): Limit image size
        - Use resize_ratio (0.5-0.8): Directly shrink images
        - Grayscale: Reduces data by 66% vs RGB
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n{'='*80}")
    print(f"PDF to Image Conversion Settings")
    print(f"{'='*80}")
    print(f"File: {pdf_path.name}")
    print()

    print("┌─────────────────────────────────┬──────────────────────────────────┐")
    print("│ Parameter                       │ Value                            │")
    print("├─────────────────────────────────┼──────────────────────────────────┤")
    print(f"│ DPI                             │ {str(dpi):32} │")
    print(f"│ Grayscale                       │ {str(grayscale):32} │")
    print(f"│ JPEG Quality                    │ {str(jpeg_quality):32} │")
    print(f"│ Resize Ratio                    │ {str(resize_ratio if resize_ratio else 'None (1.0)'):32} │")
    print(f"│ Max Width                       │ {str(max_width if max_width else 'Unlimited'):32} │")
    print(f"│ Max Height                      │ {str(max_height if max_height else 'Unlimited'):32} │")
    print(f"│ Max Pixels                      │ {str(f'{max_pixels:,}' if max_pixels else 'Unlimited'):32} │")
    print(f"│ Enhance Contrast                │ {str(enhance_contrast if enhance_contrast else 'None (1.0)'):32} │")
    print(f"│ Enhance Sharpness               │ {str(enhance_sharpness if enhance_sharpness else 'None (1.0)'):32} │")
    print(f"│ Save Directory                  │ {str(save_dir.name if save_dir else 'Not saving'):32} │")
    print("└─────────────────────────────────┴──────────────────────────────────┘")
    print()

    print("Rendering PDF...")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"  Pages rendered: {len(images)}")

    if images:
        orig_w, orig_h = images[0].width, images[0].height
        orig_pixels = orig_w * orig_h
        print(f"  Original size: {orig_w}x{orig_h} ({orig_pixels:,} pixels)")
        print()

    if grayscale:
        print(f"\nConverting to grayscale...")
        images = [img.convert("L") for img in images]

    processed = []
    for idx, img in enumerate(images, 1):
        if resize_ratio and resize_ratio != 1.0:
            new_w = int(img.width * resize_ratio)
            new_h = int(img.height * resize_ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if max_width and img.width > max_width:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)

        if max_height and img.height > max_height:
            ratio = max_height / img.height
            new_w = int(img.width * ratio)
            img = img.resize((new_w, max_height), Image.Resampling.LANCZOS)

        if max_pixels and (img.width * img.height) > max_pixels:
            ratio = (max_pixels / (img.width * img.height)) ** 0.5
            new_w = int(img.width * ratio)
            new_h = int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if enhance_contrast and enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(enhance_contrast)

        if enhance_sharpness and enhance_sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(enhance_sharpness)

        processed.append(img)

    if processed:
        final_w, final_h = processed[0].width, processed[0].height
        final_pixels = final_w * final_h

        print("Processing Results:")
        print("┌─────────────────────────────────┬──────────────────┬──────────────────┐")
        print("│ Metric                          │ Before           │ After            │")
        print("├─────────────────────────────────┼──────────────────┼──────────────────┤")
        print(f"│ Width                           │ {orig_w:16} │ {final_w:16} │")
        print(f"│ Height                          │ {orig_h:16} │ {final_h:16} │")
        print(f"│ Total Pixels                    │ {orig_pixels:16,} │ {final_pixels:16,} │")

        if final_w != orig_w or final_h != orig_h:
            reduction = (1 - final_pixels / orig_pixels) * 100
            print(f"│ Pixel Reduction                 │ {'-':16} │ {reduction:14.1f}% │")
        else:
            print(f"│ Status                          │ {'-':16} │ {'No change':16} │")

        print("└─────────────────────────────────┴──────────────────┴──────────────────┘")
        print()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving images to: {save_dir}")

        for idx, img in enumerate(processed, 1):
            img_path = save_dir / f"page_{idx}.jpg"
            img.save(img_path, "JPEG", quality=jpeg_quality, optimize=True)

        print(f"  Saved {len(processed)} images (quality={jpeg_quality})")

    return processed


def image_to_bytes(image: Image.Image, format: str = 'JPEG', quality: int = 85) -> bytes:
    """
    Convert PIL Image to bytes.

    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        quality: Quality for JPEG (0-100)

    Returns:
        Image bytes
    """
    import io
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality, optimize=True)
    buffer.seek(0)
    return buffer.getvalue()
