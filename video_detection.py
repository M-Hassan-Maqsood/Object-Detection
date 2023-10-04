#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.abspath(os.path.join('.', 'videos', 'test.mp4'))
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Couldn't open video file '{video_path}'")
    exit()

# Load the first frame
ret, frame = cap.read()

if frame is None:
    print("Error: Unable to read the first frame from the video.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load a model
# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
model = YOLO('best full human.pt')  # load a custom model

threshold = 0.5

# Create a window for displaying frames
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Video', frame)
    out.write(frame)

    # Exit the loop when 'q' is pressed or when the video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()  # Read the next frame

cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




