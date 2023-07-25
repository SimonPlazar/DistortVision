import cv2
import numpy as np

# Read the image
image = cv2.imread('screenshot.png')
image_copy = image.copy()  # make a copy of the original image

# Get image shape
h, w = image.shape[:2]

# Create zero filled binary masks
mask1 = np.zeros((h, w), np.uint8)
mask2 = np.zeros((h, w), np.uint8)

# Define center and radii of the circles
center = (w // 2, h // 2)
radius1 = min(w, h) // 4
radius2 = radius1 + 5  # make this mask slightly larger

# Draw filled circles in the masks
cv2.circle(mask1, center, radius1, (255), thickness=-1)
cv2.circle(mask2, center, radius2, (255), thickness=-1)

# Extract the circular area from the original image using the larger mask
extracted = cv2.bitwise_and(image, image, mask=mask2)

# At this point, you can manipulate 'extracted' however you like.
# Once you're done, you can put it back to the original image.
_, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
# Decode the compressed image
extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

# Replace the pixels in the original image with those from 'extracted',
# but only for the pixels within the smaller mask.
image_copy[np.where(mask1 == 255)] = extracted[np.where(mask1 == 255)]

# Display final image
cv2.imshow('Final Image', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save final image
cv2.imwrite('final_image_path.jpg', image_copy)
