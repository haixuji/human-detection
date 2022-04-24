from crypt import methods
from flask import Flask

import _init_paths
import time
import argparse
import os.path as osp

import cv2
from pose_estimation import get_pose_estimator
from tracker import get_tracker
from classifier import get_classifier
from utils.config import Config
from utils.video import Video
from utils.drawer import Drawer

from utils import utils

cfg = Config("../configs/infer_trtpose_deepsort_dnn.yaml")
pose_kwargs = cfg.POSE
clf_kwargs = cfg.CLASSIFIER
tracker_kwargs = cfg.TRACKER
args_task = 'action'

pose_estimator = get_pose_estimator(**pose_kwargs)
if args_task != 'pose':
    tracker = get_tracker(**tracker_kwargs)
    if args_task == 'action':
        action_classifier = get_classifier(**clf_kwargs)

## initiate drawer and text for visualization
drawer = Drawer(draw_numbers=False)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def action_predictor():
    action_labels = {'predictions': []}

    bgr_frame = request.get_data()
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    drawer = Drawer(draw_numbers=False)
    user_text = {
        'text_color': 'green',
        'add_blank': True,
        'Mode': 'action',
        # MaxDist: cfg.TRACKER.max_dist,
        # MaxIoU: cfg.TRACKER.max_iou_distance,
    }

    predictions = pose_estimator.predict(rgb_frame, get_bbox=True)

    if len(predictions) == 0 and args_task != 'pose':
            debug_img = bgr_frame
            tracker.increment_ages()
    else:
        # draw keypoints only if task is 'pose'
        if args_task != 'pose':
            # Tracking
            # start_track = time.time()
            predictions = utils.convert_to_openpose_skeletons(predictions)
            predictions, debug_img = tracker.predict(rgb_frame, predictions,
                                                            debug=False)
            # end_track = time.time() - start_track

            # Action Recognition
            if len(predictions) > 0 and args_task is 'action':
                predictions = action_classifier.classify(predictions)

    # add user's desired text on render image
    render_image = drawer.render_frame(bgr_frame, predictions, **user_text)

    return render_image

if __name__ == '__main__':
    app.run(debug=True, port=8888)
