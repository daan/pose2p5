import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2 as cv

from flask import Flask, Response
import json
import time
import webbrowser
from threading import Timer

import random

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

app = Flask(__name__, static_folder='p5', static_url_path='')


def get_mediapipe_coords():
    # Replace this with your actual MediaPipe logic
    # For demonstration, we'll return dummy data
    import random
    return {"x": random.random(), "y": random.random(), "z": random.random()}

def event_stream():
    while True:
        coords = get_mediapipe_coords()
        # The data must be formatted as "data: ...\n\n"
        yield f"data: {json.dumps(coords)}\n\n"
        time.sleep(1/30) # Aim for ~30 FPS


def event_stream2():
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,640) 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


        annotated_image = draw_landmarks_on_image(rgb, detection_result)

        bgr = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

        # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', bgr)

        if cv.waitKey(1) == 27:
            break
        
        coords = {"x": random.random(), "y": random.random(), "z": random.random()}
        yield f"data:{json.dumps(coords)}\n\n"


    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

@app.route('/stream')
def stream():
    # The mimetype 'text/event-stream' is crucial
    return Response(event_stream(), mimetype='text/event-stream')


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(threaded=True) # Port defaults to 5000
