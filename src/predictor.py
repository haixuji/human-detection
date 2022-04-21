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


def actionPredict(bgr_frame, rgb_frame):
    cfg = Config("../configs/infer_trtpose_deepsort_dnn.yaml")
    pose_kwargs = cfg.POSE
    clf_kwargs = cfg.CLASSIFIER
    tracker_kwargs = cfg.TRACKER
    args_task = 'action'

    ## Initiate trtpose, deepsort and action classifier
    pose_estimator = get_pose_estimator(**pose_kwargs)
    action_classifier = get_classifier(**clf_kwargs)

    ## initiate drawer and text for visualization
    drawer = Drawer(draw_numbers=False)
    user_text = {
        'text_color': 'green',
        'add_blank': True,
        'Mode': 'action',
        # MaxDist: cfg.TRACKER.max_dist,
        # MaxIoU: cfg.TRACKER.max_iou_distance,
    }

    predictions = pose_estimator.predict(rgb_frame, get_bbox=True)
    print("[INFO] prediction length: " + str(len(predictions)) + " !!!!!!!")

 
            # Tracking
            # start_track = time.time()
    predictions = utils.convert_to_openpose_skeletons(predictions)
    print("[INFO] skeleton prediction lenght: " + str(len(predictions)) + " !!!!!!!")
    # predictions, debug_img = tracker.predict(rgb_frame, predictions,
    #                                                 debug=True)
    # end_track = time.time() - start_track

    # Action Recognition
    if len(predictions) > 0 and args_task == 'action':
        predictions = action_classifier.classify(predictions)
        print("[INFO] action prediction", predictions)
        for pred in predictions:
            print("[pred-action]", pred.action)
                # if pred.action[0]:
                #     action_label = '{}: {:.2f}'.format(*pred.action) if pred.action[0] else ''
                #     print("[action label]" + action_label + " -------------")

    # add user's desired text on render image
    # render_image = drawer.render_frame(bgr_frame, predictions, **user_text)

    # return render_image

