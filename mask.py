import cv2
import numpy as np
import time

def compress_and_overlay(image, mask, quality):
    # Make a copy of the original image
    image_copy = image.copy()

    # Create a mask slightly larger than the input mask
    dilated_mask = cv2.dilate(mask, None, iterations=5)

    # Extract the area defined by the input mask from the original image
    extracted = cv2.bitwise_and(image, image, mask=dilated_mask)

    # Compress the extracted area
    _, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    # Decode the compressed image
    extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

    # Replace the pixels in the original image with those from 'extracted',
    # but only for the pixels within the input mask.
    image_copy[np.where(mask == 255)] = extracted[np.where(mask == 255)]

    return image_copy


# Example usage:
if __name__ == "__main__":
    # Read the image
    image = cv2.imread('input/screenshot.png')

    # Create a binary mask (you can generate masks for any shape you need)
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    # Example: create a circular mask
    # center = (image.shape[1] // 2, image.shape[0] // 2)
    # radius = min(image.shape[1], image.shape[0]) // 4
    # cv2.circle(mask, center, radius, (255), thickness=-1)

    x, y, w, h = image.shape[1] // 4, image.shape[0] // 4, image.shape[1] // 2, image.shape[0] // 2
    mask[y:y + h, x:x + w] = 255

    # Call the compress_and_overlay function with the image and mask
    start = time.time()
    final_image = compress_and_overlay(image, mask, 5)
    print("Time taken: {:.2f} seconds".format(time.time() - start))

    # Display the final image
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the final image
    cv2.imwrite('output/final_image_path.jpg', final_image)
