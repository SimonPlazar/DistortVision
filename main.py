from PIL import Image
import numpy as np

def rgb_to_ycbcr(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr

def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return int(round(r)), int(round(g)), int(round(b))

def compress_half_image(image_path, quality):
    if quality <= 1:
        raise ValueError("Quality must be greater than 1.")

    # Load the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (if it's not already)
    image = image.convert("RGB")

    # Get image dimensions
    width, height = image.size

    # Create a copy of the image to leave half of it uncompressed
    compressed_image = image.copy()

    # Loop over the bottom half of the image and perform compression
    for y in range(height // 2, height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))

            # Convert RGB to YCbCr
            y, cb, cr = rgb_to_ycbcr(r, g, b)

            # Perform compression (e.g., DCT and quantization) on Y, Cb, and Cr components
            # For simplicity, we will only divide each value by the quality factor
            compressed_y = y // quality
            compressed_cb = cb // quality
            compressed_cr = cr // quality

            # Convert back from YCbCr to RGB
            r_compressed, g_compressed, b_compressed = ycbcr_to_rgb(compressed_y, compressed_cb, compressed_cr)

            # Ensure the RGB values are within the valid range of 0 to 255
            r_compressed = max(0, min(255, r_compressed))
            g_compressed = max(0, min(255, g_compressed))
            b_compressed = max(0, min(255, b_compressed))

            # Assign the compressed pixel to the compressed_image
            compressed_image.putpixel((x, y), (r_compressed, g_compressed, b_compressed))

    # Show the original and compressed images (optional)
    image.show()
    compressed_image.show()

if __name__ == "__main__":
    # Specify the image path and the desired compression level (2 to 100)
    image_path = "input/screenshot.png"
    compression_level = 50  # Adjust this value for different levels of compression

    compress_half_image(image_path, compression_level)
