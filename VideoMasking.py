import cv2
import numpy as np
import torch
from ultralytics import YOLO
import cvzone
from scipy import ndimage as ndi

def get_squere_mask(width, height):
    # Create a binary square mask (you can change this according to your needs)
    mask = np.zeros((height, width), np.uint8)
    x, y, w, h = width // 4, height // 4, width // 2, height // 2
    mask[y:y + h, x:x + w] = 255
    return mask

def compress_and_overlay(image, mask, quality):
    # Create a mask slightly larger than the input mask
    dilated_mask = cv2.dilate(mask, None, iterations=5)

    # Extract the area defined by the larger mask from the original image
    extracted = cv2.bitwise_and(image, image, mask=dilated_mask)

    # Compress the extracted area
    _, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    # Decode the compressed image
    extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

    # Replace the pixels in the original image with those from 'extracted',
    # but only for the pixels within the smaller mask.
    image[np.where(mask == 255)] = extracted[np.where(mask == 255)]

    return image


if __name__ == "__main__":
    # Load the model
    model = YOLO("dnn\\yolov8x-seg.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # MAGIC!!!
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fpsReader = cvzone.FPS()

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Get the masks from the model
        results = model.predict(source=frame, conf=0.59, classes=0, verbose=False, device=0, retina_masks=True)[0].masks
        # results = model.predict(source=cv2.resize(frame, (640, 480)), conf=0.59, classes=0, verbose=False, device=0, retina_masks=True)[0].masks

        if results is not None: # If a mask is found
            # mask = np.multiply(results.data[0].numpy(), 255).astype(np.uint8) # First mask
            mask = (results.data[0] * 255).to(torch.uint8)
            # mask = torch.mul(results.data[0], 255).to(torch.uint8)

            for i in range(1, results.data.shape[0]): # For each other mask
                #mask = np.bitwise_or(mask, np.multiply(results.data[i].numpy(), 255).astype(np.uint8)) # Combine the masks
                mask = torch.bitwise_or(mask, (results.data[i] * 255).to(torch.uint8)) # Combine the masks
                # mask = torch.bitwise_or(mask, torch.mul(results.data[0], 255).to(torch.uint8)) # Combine the masks

            # Pass the mask to cpu memory
            mask = mask.cpu().numpy()

            # Resize the mask to the video stream size
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create dilated mask and extract the area defined by the mask from the original image
            dilated_mask = cv2.dilate(mask, None, iterations=5)
            # dilated_mask = ndi.binary_dilation(mask, None, iterations=5)
            extracted = cv2.bitwise_and(frame, frame, mask=dilated_mask)

            # Alter extracted image

            # extracted = np.full((height, width, 3), (203,192,255), dtype=np.uint8) # Pink
            # extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV) # Convert to HSV
            _, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), 6])
            extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

            # Replace the pixels in the original image with those from 'extracted',
            # frame[np.where(mask == 255)] = extracted[np.where(mask == 255)]

            # Use boolean indexing to replace the pixels in the original image
            # mask = mask == 255
            # frame[mask] = extracted[mask]

            # Use bitwise operations to replace the pixels in the original image
            roi = cv2.bitwise_and(extracted, extracted, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

            frame = cv2.add(background, roi)

        else: # If no mask is found
            cv2.putText(frame, "No mask found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Display a message

        # Display the processed frame with FPS information
        fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
        cv2.imshow('Processed Frame', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()
