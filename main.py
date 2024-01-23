import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
WEIGHTS_PATH = '/yolo-coco/yolov3.weights'
CFG_PATH = '/yolo-coco/yolov3.cfg'
NAMES_PATH = '/yolo-coco/coco.names'
VIDEO_PATHS = ['/Users/thanhle/Desktop/pythonProject/Entrance test 2/Task 1/Videos/Video 1.mp4',
               '/Users/thanhle/Desktop/pythonProject/Entrance test 2/Task 1/Videos/Video 2.mp4',
               '/Users/thanhle/Desktop/pythonProject/Entrance test 2/Task 1/Videos/Video 3.mp4']
path= ['/Users/thanhle/Desktop/pythonProject/Entrance test 2/Task 1/Videos/Video 3.mp4']
PERSON_TEMPLATE_PATH = '/Users/thanhle/Desktop/pythonProject/Entrance test 2/Task 1/Input/Level 3.jpg'
# Load YOLO
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers().reshape(-1).tolist()
output_layers = [layer_names[i - 1] for i in output_layer_indices]
with open(NAMES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def lvl1():
    OUTPUT_DIR = '/Users/thanhle/Desktop/pythonProject/output_new/lvl1'
    for idx, video_path in enumerate(VIDEO_PATHS):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_count / fps
            print(f"Timestamp: {timestamp:.2f} seconds")

            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Information to show on the screen (class id, confidence, bounding box coordinates)
            class_ids = []
            confidences = []
            boxes = []

            # For each detected object
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    label = str(classes[class_ids[i]])
                    if label == 'truck':
                        x, y, w, h = boxes[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        save_path = os.path.join(OUTPUT_DIR, f"Video {idx + 1}", "Object X", f"Frame {frame_count}.jpg")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, frame)

            frame_count += 1
        cap.release()

    print("Detection completed!")

def lvl2():
    OUTPUT_DIR = '/Users/thanhle/Desktop/pythonProject/output_new/lvl2'
    for idx, video_path in enumerate(VIDEO_PATHS):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            timestamp = frame_count / fps
            #print(f"Timestamp: {timestamp:.2f} seconds")
            if not ret or timestamp>20:
                break
            #timestamp = frame_count / fps
            #print(f"Timestamp: {timestamp:.2f} seconds")
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == 'truck':
                        center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                        x, y = int(center_x - w / 2), int(center_y - h / 2)

                        # Check color of the detected truck
                        roi = frame[y:y+h, x:x+w]
                        if roi.size==0:
                            continue
                        # Convert the ROI to HSV
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        # Define two ranges for red color
                        lower_red1 = np.array([0, 70, 50])
                        upper_red1 = np.array([10, 255, 255])

                        lower_red2 = np.array([170, 70, 50])
                        upper_red2 = np.array([180, 255, 255])


                        # Create masks for the red regions
                        red_mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
                        red_mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)

                        # Combine the two masks
                        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

                        # Check if a significant red region is present in the ROI
                        if np.sum(red_mask)/255 > 0.1 * roi.size:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            save_path = os.path.join(OUTPUT_DIR, f"Video {idx + 1}", "Object X", f"Frame {frame_count}.jpg")
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            cv2.imwrite(save_path, frame)




            frame_count += 1
        cap.release()

    print("Red Truck Detection Completed!")


def lvl3(path):
    OUTPUT_DIR = '/Users/thanhle/Desktop/pythonProject/output_new/lvl3'
    template = cv2.imread(PERSON_TEMPLATE_PATH, 0)
    h, w = template.shape[:2]

    # Process videos
    for idx, video_path in enumerate(path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            timestamp = frame_count / fps
            print(f"Timestamp: {timestamp:.2f} seconds")
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply template matching
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(max_val)
            threshold = 0.8  # Threshold for a "good" match;
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):  # Iterate through the detected locations
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

                save_path = os.path.join(OUTPUT_DIR, f"Video 3", "Object X", f"Frame {frame_count}.jpg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, frame)

            frame_count += 1
        cap.release()

    print("Person Detection Completed!")