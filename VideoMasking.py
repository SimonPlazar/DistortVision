import cv2
import numpy as np
import time

def compress_and_overlay(image, mask, quality):
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
    image[np.where(mask == 255)] = extracted[np.where(mask == 255)]

    return image


if __name__ == "__main__":
    # Create the video capture object (replace '0' with the appropriate video source)
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_num = cap.get(cv2.CAP_PROP_FPS)

    print(fps_num)

    # Create a binary square mask (you can change this according to your needs)
    mask = np.zeros((height, width), np.uint8)
    x, y, w, h = width // 4, height // 4, width // 2, height // 2
    mask[y:y + h, x:x + w] = 255

    start_time_fps = time.time()
    frame_counter = 0
    fps = 0

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Measure the processing time of the function
        start_time = time.time()
        frame = compress_and_overlay(frame, mask, 5)
        print("Time taken: {:.2f} seconds".format(time.time() - start_time))

        # Calculate and display FPS
        frame_counter += 1
        if (time.time() - start_time_fps) > 1:
            fps = frame_counter / (time.time() - start_time_fps)
            frame_counter = 0
            start_time_fps = time.time()

        # Display the processed frame with FPS information
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Processed Frame', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()
