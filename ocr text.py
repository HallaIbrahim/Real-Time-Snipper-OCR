import cv2
import numpy as np
import pyautogui
import pytesseract
import time

# Set the duration for recording in seconds
duration = 120

# Set the frame capture interval in seconds
# frame_interval = 2

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    # Capture screenshot and convert to a numpy array
    screenshot = pyautogui.screenshot()
    screenshot_array = np.array(screenshot)

    # Set the region of interest coordinates (x, y, width, height)
    region = cv2.selectROI("Select Region of Interest", screenshot_array)

    # Create a VideoWriter object to save the recorded frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('recorded_video.mp4', fourcc, 20.0, (int(region[2]), int(region[3])))

    # Start the recording
    start_time = time.time()

    while (time.time() - start_time) < duration:
        # Capture screenshot
        frame = np.array(pyautogui.screenshot())

        # Crop to the region of interest
        cropped_frame = frame[int(region[1]):int(region[1] + region[3]), int(region[0]):int(region[0] + region[2])]

        # Preprocessing
        # gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
        # dilate = cv2.dilate(thresh, kernel, iterations=1)

        # Perform OCR to get bounding boxes and characters
        boxes = pytesseract.image_to_boxes(cropped_frame, lang='eng', config='--psm 6 --oem 1')
        himg, wimg, _ = cropped_frame.shape

        # Iterate over the boxes and draw them on the frame
        for box in boxes.splitlines():
            buffer=10
            box = box.split(' ')
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            x += 60
            #cv2.rectangle(frame, (x-buffer, himg - y+120), (x-buffer, himg - y+120), (0, 255, 0), 3)
            cv2.putText(frame, box[0], (x, himg - y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    # Print the recognized character
            #print("Character:", box[0])
        # Write the frame to the output video file
        output.write(frame)

        # Display the frame
        cv2.imshow("Recording", frame)
        if cv2.waitKey(0) == ord('q'):
            break

        # Wait for the specified frame interval
        # time.sleep(frame_interval)

    # Release the VideoWriter and close windows
    output.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", e)