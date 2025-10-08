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
from threading import Timer, Thread, Lock
import queue

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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


# Live reload functionality
class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # When a file in the p5 directory is modified, send a reload message
        # to all connected clients.
        with listeners_lock:
            for q in listeners:
                # Use non-blocking put to avoid waiting if a queue is full
                try:
                    q.put_nowait('reload')
                except queue.Full:
                    pass

def file_watcher():
    path = './p5'
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


coords_queue = queue.Queue()
app = Flask(__name__, static_folder='p5', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disable caching for development

# For live reload
listeners = []
listeners_lock = Lock()




def run_mediapipe(q):
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

        if detection_result.pose_landmarks:
            # Send all landmarks
            landmarks = detection_result.pose_landmarks[0]
            coords = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
            q.put(coords)
        else:
            # Send empty list if no pose detected
            q.put([])

        if cv.waitKey(1) == 27:
            break
        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def event_stream():
    while True:
        coords = coords_queue.get()
        yield f"data: {json.dumps(coords)}\n\n"

@app.route('/stream')
def stream():
    # The mimetype 'text/event-stream' is crucial
    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/events')
def events():
    def gen():
        q = queue.Queue()
        with listeners_lock:
            listeners.append(q)
        try:
            while True:
                msg = q.get()
                yield f'data: {msg}\n\n'
        finally:
            with listeners_lock:
                listeners.remove(q)
    return Response(gen(), mimetype='text/event-stream')


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/index.html")


if __name__ == "__main__":
    # Start Flask in a background thread.
    # The `daemon=True` flag means the thread will exit when the main thread exits.
    flask_thread = Thread(target=lambda: app.run(threaded=True, use_reloader=False), daemon=True)
    flask_thread.start()

    # Start the file watcher in another background thread.
    watcher_thread = Thread(target=file_watcher, daemon=True)
    watcher_thread.start()

    Timer(1, open_browser).start()
    
    # Run MediaPipe in the main thread, as required by OpenCV's GUI functions.
    run_mediapipe(coords_queue)
