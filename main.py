from PIL import Image, ImageChops
import numpy as np

def jpeg_artifact(image_path, quality):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (if it's not already)
    image = image.convert("RGB")

    # Save a copy of the original image for comparison later
    original_image = image.copy()

    # Save the image with JPEG compression
    image.save("compressed.jpg", format="JPEG", quality=quality)

    # Load the compressed image back
    compressed_image = Image.open("compressed.jpg")

    # Show the images (optional)
    original_image.show()
    compressed_image.show()

if __name__ == "__main__":
    # Specify the image path and the desired JPEG quality (1 to 95)
    image_path = "screenshot.png"
    jpeg_quality = 100

    jpeg_artifact(image_path, jpeg_quality)