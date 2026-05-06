import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("Rules:")
print("- 's' screenshot saved")
print("- 'q' program closed")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.3)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("CodeAlpha: Object Detection & Tracking", annotated_frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF

        # Save screenshot if 's' is pressed
        if key == ord('s'):
            cv2.imwrite("detection_screenshot.jpg", annotated_frame)
            print("Success: Screenshot saved as detection_screenshot.jpg")

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break
    else:
        # Break the loop if there's an issue with the video capture
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
