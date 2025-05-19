import sys
import os
import load_config as config
import tracker_evaluation as tracker_evaluation 
from tracker.deepsort.deepsort.tracker import DeepSortTracker
from tracker.deepsort.deepsort.detection import Detection
import numpy as np
sys.path.append(os.path.abspath("../tracker/deep_sort_real_time"))
from tracker.deep_sort_real_time.deep_sort_real_time.deepsort_tracker import DeepSort
sys.path.append(os.path.abspath("../tracker/strongsort/strongsort"))
from tracker.strongsort.strongsort.strong_sort import StrongSORT
sys.path.append(os.path.abspath("../detector_tracker/ultralytics/trackers"))
from detector_tracker.ultralytics.trackers.byte_tracker import BYTETracker
from detector_tracker.ultralytics.trackers.bot_sort import BOTSORT
import argparse
import os.path as osp
import time
import cv2
import torch



#--------------------------------- LOAD MODEL WEIGH ----------------------------------


#-------------------------------- DEEP_SORT -----------------------------------------

# def to_xywh(box):
#   x = float(box.xywh.cpu().numpy()[0][0])
#   y = float(box.xywh.cpu().numpy()[0][1])
#   w = float(box.xywh.cpu().numpy()[0][2])
#   h = float(box.xywh.cpu().numpy()[0][3])
#   return [x,y,w,h]

# def to_conf(box):
#   return box.conf.cpu().numpy()[0]

# def to_cls(box):
#   return box.cls.cpu().numpy()[0]

# def convertToDectections(results):
#   detection = []
#   for result in results:
#     for i in result.boxes:
#       xywh = to_xywh(i)
#       conf = to_conf(i)
#       cls = to_cls(i)
#       detection.append(Detection(xywh, conf, cls))
#   return detection



# def loadModelDeepSortTracker():
#   loadConfig = config.yaml_load()
#   max_iou_distance = loadConfig.get("deep_sort")['max_iou_distance']
#   max_age = loadConfig.get("deep_sort")['max_age']
#   n_init = loadConfig.get("deep_sort")['n_init']
#   max_dist = loadConfig.get("deep_sort")['max_dist']
#   nn_budget = loadConfig.get("deep_sort")["nn_budget"]
#   deepSortTracker = DeepSortTracker(max_iou_distance = max_iou_distance,
#                                     max_age = max_age,
#                                     n_init = n_init,
#                                     max_dist = max_dist,
#                                     nn_budget = nn_budget)
#   return deepSortTracker


# def deepSortTrackerUpdate(model,detections):
#   return model.update(detections)

#-------------------------------- DEEP_SORT ---------------------------------------------
#########################################################################################
#-------------------------------- DEEP_SORT_PIP -----------------------------------------

# def loadModel():
#   loadConfig = config.yaml_load()
#   max_iou_distance = loadConfig.get("deep_sort")['max_iou_distance']
#   max_age = loadConfig.get("deep_sort")['max_age']
#   n_init = loadConfig.get("deep_sort")['n_init']
#   max_dist = loadConfig.get("deep_sort")['max_dist']
#   nn_budget = loadConfig.get("deep_sort")["nn_budget"]
#   # embedder = loadConfig.get("deep_sort")["embedder"]
#   return DeepSort(max_iou_distance = max_iou_distance,
#                   max_age = max_age,
#                   n_init = n_init,
#                   max_cosine_distance = max_dist,
#                   nn_budget = nn_budget)


# def deepSortTrackerUpdate(model,raw_detections,frame):
#   return model.update_tracks(raw_detections = raw_detections, frame = frame)

# def handleResultTracker(tracks):
#   for track in tracks:
#     if not track.is_confirmed():
#         continue
#     track_id = track.track_id
#     ltrb = track.to_ltrb()
#     conf = track.get_det_conf()

#-------------------------------- DEEP_SORT_PIP -----------------------------------------
#########################################################################################
#-------------------------------- STRONG_SORT_PIP -----------------------------------------


# def loadModel():
#   loadConfig = config.yaml_load()
#   fp16 = loadConfig.get("strong_sort")['fp16']
#   device = loadConfig.get("strong_sort")['device']
#   max_iou_distance = loadConfig.get("strong_sort")['max_iou_distance']
#   max_age = loadConfig.get("strong_sort")['max_age']
#   n_init = loadConfig.get("strong_sort")['n_init']
#   max_dist = loadConfig.get("strong_sort")['max_dist']
#   nn_budget = loadConfig.get("strong_sort")["nn_budget"]
#   model_weights = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/tracker/model_weight/osnet_x0_25.pt"
#   return StrongSORT(model_weights = model_weights,
#                   device = device,
#                   fp16 = fp16,
#                   max_iou_distance = max_iou_distance,
#                   max_age = max_age,
#                   n_init = n_init,
#                   max_dist = max_dist,
#                   nn_budget = nn_budget)

# # Detetion [x;y;x;y;conf;clss]
# def updateStrongSort(model,dets,ori_img):
#   return model.update(dets = dets, ori_img = ori_img)


# def handleResults():
#   pass


#-------------------------------- STRONG_SORT_PIP -----------------------------------------
#########################################################################################
#-------------------------------- BYTE_TRACKER -----------------------------------------

# def loadConfig():
#   return config.yaml_load().get("byte_track")

# def loadArgs():
#     config = loadConfig()
#     parser = argparse.ArgumentParser("Tracker args!")
#     parser.add_argument("--fuse_score",dest="fuse_score",default=config['fuse_score'],action="store_true",help="Fuse conv and bn for testing.",)
#     parser.add_argument("--track_high_thresh", type=float, default=config['track_high_thresh'], help="track high thresh")
#     parser.add_argument("--track_low_thresh", type=float, default=config['track_low_thresh'], help="track_low_thresh")
#     parser.add_argument("--new_track_thresh", type=float, default=config['new_track_thresh'], help="track_low_thresh")
#     parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
#     parser.add_argument("--track_buffer", type=int, default=config['track_buffer'], help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=config['match_thresh'], help="matching threshold for tracking")
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     return parser.parse_args()


# def loadModel():
#   return BYTETracker(args = loadArgs())
  
# def updateByteTracker(model, results, img):
#   return model.update(results = results, img = img )



#-------------------------------- BYTE_TRACKER -----------------------------------------
#########################################################################################
#-------------------------------- BOT_SORT -----------------------------------------


def loadConfig():
  return config.yaml_load().get("bot_sort")


def loadArgs():
    config = loadConfig()
    parser = argparse.ArgumentParser("Tracker args!")
    parser.add_argument("--fuse_score",dest="fuse_score",default=config['fuse_score'],action="store_true",help="Fuse conv and bn for testing.",)
    parser.add_argument("--track_high_thresh", type=float, default=config['track_high_thresh'], help="track high thresh")
    parser.add_argument("--track_low_thresh", type=float, default=config['track_low_thresh'], help="track_low_thresh")
    parser.add_argument("--new_track_thresh", type=float, default=config['new_track_thresh'], help="track_low_thresh")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=config['track_buffer'], help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=config['match_thresh'], help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--appearance_thresh', type=float, default=config['appearance_thresh'], help='appearance thresh')
    parser.add_argument('--proximity_thresh', type=float, default=config['proximity_thresh'], help='proximity thresh')
    parser.add_argument('--with_reid', type=float, default=config['with_reid'], help='with reid')
    parser.add_argument('--gmc_method', type=str, default=config['gmc_method'], help='gmc_method')
    return parser.parse_args()



def loadModel():
  return BOTSORT(args = loadArgs())
  
def updateBotSort(model, dets, scores,cls, img):
  trackResult = model.init_track(dets = dets, scores = scores, cls = cls, img = img)
  return model.multi_predict(tracks = trackResult)









#-------------------------------- BOT_SORT -----------------------------------------



