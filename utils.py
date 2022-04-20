import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
import cv2
import numpy as np
import glob

# ============================= Counter Imports =============================
import mediapipe as mp


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_counts(exercise, path):
    #mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if exercise == "push_up":
        return counter_push_up(path, mp_pose)

    if exercise == "squat" or exercise == "lunge":
        return counter_squat(path, mp_pose)

    return 0, 0


# ==================================================================

def counter_squat(path, mp_pose):

    cap = cv2.VideoCapture(path)

    # Variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()

                #w = frame.shape[1]
                #h = 480
                #w = int((h/frame.shape[1])*w)
                frame = cv2.resize(frame, (640, 480))

                #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                # Color frame is RGB in mediapipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

                    angle = calculate_angle(shoulder, hip, knee)

                    # Counter logic
                    if angle > 145:
                        stage = 'up'

                    if angle < 125 and stage == 'up':
                        stage = 'down'
                        counter += 1
                        print(counter)

                except:
                    pass

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()

    calories = 0.32*counter
    return counter, calories
# =========================================================================================


# =======================================================================================
def counter_push_up(path, mp_pose):
    # ========================================================================================

    cap = cv2.VideoCapture(path)

    # Variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()

                #w = frame.shape[1]
                #h = 480
                #w = int((h/frame.shape[1])*w)
                frame = cv2.resize(frame, (640, 480))

                #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                # Color frame is RGB in mediapipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Counter logic
                    if angle > 145:
                        stage = 'up'

                    if angle < 125 and stage == 'up':
                        stage = 'down'
                        counter += 1
                        print(counter)

                except:
                    pass

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()

    calories = 0.32*counter
    return counter, calories
# =========================================================================================


def load_model():

    device = "cpu"

    # Pick a pretrained model and load the pretrained weights
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo",
                           model=model_name, pretrained=True)

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()

    return model, device


def get_class_names():
    with open("kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    return kinetics_id_to_classname


####################
# SlowFast transform
####################

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def get_transform():
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )
    return transform


# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second


def get_video_as_inputs(video_path, transform, device):
    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"][:120]
    inputs = [i.to(device)[None, ...] for i in inputs]

    return inputs


def get_preds(inputs, model, kinetics_id_to_classname):
    preds = model(inputs)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=2).indices

    # Map the predicted classes to the label names
    pred_class_names = [
        kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
    print("Predicted labels: %s" % ", ".join(pred_class_names))

    return pred_class_names
