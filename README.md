# pose2p5


# install

## 

1. clone the github archive. 
2. create the virtual environment
```
uv sync. 
```

3. download models into the models folder
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

4. run
```
python main.py
```
starts the default camera and streams the mediapipe coordinates to p5 using Server Side Events.


