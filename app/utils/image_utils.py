"""
Image processing utilities.
Handles image compression and optimization.
"""

import io
from pathlib import Path
from PIL import Image


def compress_image(
    file_path: Path,
    mime_type: str,
    size_threshold: int = 500 * 1024,
    max_dimension: int = 2048,
    quality: int = 85
):
    """
    Compress image in-place if it exceeds size threshold.
    Resizes large images and re-encodes with quality tuning.
    Thresholds: 500 KB soft limit, 1 MB aggressive limit.
    
    Args:
        file_path: Path to the image file
        mime_type: MIME type of the image
        size_threshold: Size threshold in bytes (default: 500 KB)
        max_dimension: Maximum width or height (default: 2048)
        quality: JPEG/WebP quality 1-100 (default: 85)
    
    Note: Still fails for some images. Say 5 MB, 4096 x 4096 image. Need to fix later.
    """
    try:
        file_size = file_path.stat().st_size
        if file_size < size_threshold:
            return  # No compression needed
        
        img = Image.open(file_path)
        original_mode = img.mode
        
        # Convert RGBA/LA/P to RGB if saving as JPEG (JPEG doesn't support transparency)
        if img.mode in ("RGBA", "LA", "P"):
            if mime_type in ("image/jpeg", "application/octet-stream"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
        
        # Resize if dimensions are too large
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # Determine output format and quality
        if mime_type == "image/png":
            output_format = "PNG"
            save_kwargs = {"optimize": True}
        elif mime_type == "image/webp":
            output_format = "WEBP"
            save_kwargs = {"quality": quality}
        else:  # Default to JPEG for unknown/JPEG types
            output_format = "JPEG"
            save_kwargs = {"quality": quality, "optimize": True}
        
        # Save to in-memory buffer first to check size
        buffer = io.BytesIO()
        img.save(buffer, format=output_format, **save_kwargs)
        compressed_size = buffer.tell()
        
        # Only write back if compressed size is smaller
        if compressed_size < file_size:
            buffer.seek(0)
            file_path.write_bytes(buffer.getvalue())
    
    except Exception as e:
        # Log error but don't fail upload if compression fails
        print(f"Warning: Image compression failed for {file_path.name}: {e}")
