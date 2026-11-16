What does that project do? 
- Trains a pupil detection model on the Ev-eye dataset
- The model predicts the pupil mask
- An ellipse is fitted on the predicted mask to retrieve ellipse parameters and pupil center
- A mathematical model uses 2d ellipses to estimate 3d gaze
- Given a video, an annotation pipeline produces an output video with annotated gaze vector

<img width="717" height="388.67" alt="image" src="https://github.com/user-attachments/assets/98401edb-4fb2-457f-a3d3-feb5e760bd2f" />

How to use this repo
1. To train the model, make sure to have downloaded the "Data_davis_labelled_with_mask" folder from the official Ev-Eye repo
2. Use config.json to setup the exeperiment, do not change H,W if you use the ev-eye dataset
3. run: training.py ,to train the model, modify BASE_FILTER in model.py to choose the dimension of th emodel
4. Training finishes with an overall test of the trained model on the choosen test_session
5. run: tests_only , to re-test a single model checkpoint
6. to predict gaze on a video put the video in the project folder in .mov format, then run:
     annotate_video.py "eye_recording.mov" "_/best.keras" "output.mp4"


<img width="420.33" height="310" alt="image" src="https://github.com/user-attachments/assets/691f320c-ae91-4c7e-94a0-3ca42851646d" />

Results:
- On the Ev-Eye dataset both big and small (from 2M parameters to 17k params) perform very well: > 90% IOU
- When predicting gaze in a video bigger models seem to be preferrable. 
