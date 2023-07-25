import cv2
import numpy as np

# Read the image
image = cv2.imread('screenshot.png')

# Get image shape
h, w = image.shape[:2]

# Create zero filled binary mask
mask = np.zeros((h, w), np.uint8)

# Define center and radius of the circle
center = (w // 2, h // 2)
radius = min(w, h) // 4

# Draw filled circle in the mask
cv2.circle(mask, center, radius, (255), thickness=-1)

# Extract the circle area from the original image
extracted = cv2.bitwise_and(image, image, mask=mask)

# At this point you can manipulate 'extracted' however you like.
# Once you're done, you can put it back to the original image.
_, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
# Decode the compressed image
extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

# Create an inverse mask
inverse_mask = cv2.bitwise_not(mask)

# Clear the circular area in the original image
image_cleared = cv2.bitwise_and(image, image, mask=inverse_mask)

# Now combine the manipulated 'extracted' with 'image_cleared'
final_image = cv2.bitwise_or(image_cleared, extracted)

# Display final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save final image
cv2.imwrite('final_image_path.jpg', final_image)
