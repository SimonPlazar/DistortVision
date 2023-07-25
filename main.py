from PIL import Image
import numpy as np
import random

def simulate_compression_artifacts(image_path, quality):
    distortion = ((100 - quality)/2).__floor__()

    # Load the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (if it's not already)
    image = image.convert("RGB")

    # Save a copy of the original image for comparison later
    original_image = image.copy()

    # Get image dimensions
    width, height = image.size

    # Randomly distort pixels to simulate compression artifacts
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            noise_r = random.randint(-distortion, distortion)
            noise_g = random.randint(-distortion, distortion)
            noise_b = random.randint(-distortion, distortion)
            new_r = max(0, min(255, r + noise_r))
            new_g = max(0, min(255, g + noise_g))
            new_b = max(0, min(255, b + noise_b))
            image.putpixel((x, y), (new_r, new_g, new_b))

    # Save the final distorted image
    image.save("compressed.jpg", format="JPEG", quality=quality)

    compressed_image = Image.open("compressed.jpg")

    # Show the original and distorted images (optional)
    original_image.show()
    compressed_image.show()

if __name__ == "__main__":
    # Specify the image path and the desired compression level (0 to 100)
    image_path = "screenshot.png"
    compression_level = 4  # Adjust this value for different levels of distortion

    simulate_compression_artifacts(image_path, compression_level)