from flask import Flask,render_template,request, redirect, url_for,flash,json,session,jsonify,Response
import os
import cv2
import ssl
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from threading import Thread
ssl._create_default_https_context = ssl._create_unverified_context


app = Flask(__name__)
app.secret_key = "super secret key"
UPLOAD_FOLDER = "static/uploads"
OUTPUT_DIR = "static/output_frames"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run=False

def process_video_faster_rcnn(video_path):
    # Load the pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # If you have a GPU, use it for processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = T.Compose([T.ToTensor()])

    cap = cv2.VideoCapture(video_path)
    analytics_results = []
    global run
    while True:
        ret, frame = cap.read()

        if not ret or not run:
            break
        # Convert frame to tensor and make prediction
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(frame_tensor)

        # Extract the bounding boxes and labels from the prediction
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()


        # Filter out low confidence detections and detections that aren't "person"
        selected_indices = (scores >= 0.5) & (labels == 1)
        selected_boxes = boxes[selected_indices]

        frame_data = {
            "people_count": int(len(selected_boxes)),  # assuming label 1 is for 'person'
            "objects": []
        }

        for box in zip(selected_boxes):
            if not run:
                break
            else:
                frame_data["objects"].append({
                    "label": 1,
                    "box": list(box)
                })


        analytics_results.append(frame_data)
    cap.release()

    return analytics_results
def numpy_to_python(value):
    if isinstance(value, (list, tuple)):
        return [numpy_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: numpy_to_python(v) for k, v in value.items()}
    if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(value)
    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.ndarray,)):  # handle ndarray
        return value.tolist()
    return value
@app.route("/",methods=['GET','POST'])
def index():
    analytics = None
    video_path=None
    if request.method=='POST':
        global run
        run = True
        video_file = request.files['file']
        if (video_file==''):
            return render_template('index.html',error="Please upload video")
        else:
            if video_file:
                video_path = "static/uploads/"+video_file.filename
                video_file.save(video_path)

                # Process the video
                raw_analytics = process_video_faster_rcnn(video_path)

                # Convert all numpy types to Python native types
                analytics = numpy_to_python(raw_analytics)
    return render_template('index.html',analytics=analytics,video=video_path)

@app.route("/terminate",methods=['GET'])
def terminate():
    global run
    run = False
    return jsonify(terminate='ok')

if __name__=="__main__":
    app.run(host='127.0.0.0',port=8000,debug=True)