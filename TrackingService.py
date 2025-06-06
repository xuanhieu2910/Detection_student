import math
import sys
import os
import time

import numpy as np
import load_config as config
from detector_tracker.ultralytics.utils.metrics import bbox_ioa

sys.path.append(os.path.abspath("\\tracker\\deep_sort_real_time"))
from tracker.deep_sort_real_time.deep_sort_real_time.deepsort_tracker import DeepSort
sys.path.append(os.path.abspath("\\tracker\\strongsort\\strongsort"))
from tracker.strongsort.strongsort.strong_sort import StrongSORT
sys.path.append(os.path.abspath("\\detector_tracker\\ultralytics\\trackers"))
from detector_tracker.ultralytics.trackers.byte_tracker import BYTETracker
from detector_tracker.ultralytics.trackers.bot_sort import BOTSORT
import argparse
import cv2
import torch
import ComparetiveService as cs
import torch.nn.functional as F
from torchvision import ops

class TrackingService:


  """
  array [xyxy, xywh, conf, cls, extraction, id, max_age]
  """
  DETECTIONS_STORES = []
  MAX_AGE = 30
  INIT_MAX_AGE = 1
  MATCH_THRESHOLD = 0.8


  def __init__(self, typeModelTracking, run_original):
    self.run_original = run_original
    self.type_model = typeModelTracking
    self.model = self.handleInitModelTracking()
    self.comparativeService = cs.ComparetiveService()




  def handleInitModelTracking(self):
    if self.type_model == "DeepSort":
      return self.loadModelDeepSort()
    if self.type_model == "StrongSort":
      return self.loadModelStrongSort()
    if self.type_model == "ByteTracker":
      return self.loadModelByteTracker()
    return None


  def loadModelDeepSort(self):
    loadConfig = config.yaml_load()
    max_iou_distance = loadConfig.get("deep_sort")['max_iou_distance']
    max_age = loadConfig.get("deep_sort")['max_age']
    n_init = loadConfig.get("deep_sort")['n_init']
    max_dist = loadConfig.get("deep_sort")['max_dist']
    nn_budget = loadConfig.get("deep_sort")["nn_budget"]
    embedder = loadConfig.get("deep_sort")["embedder"]
    embedder_model_name =  loadConfig.get("deep_sort")["embedder_model_name"]
    embedder_wts =  loadConfig.get("deep_sort")["embedder_wts"]
    self.MAX_AGE = max_age
    self.INIT_MAX_AGE = n_init
    return DeepSort(max_iou_distance = max_iou_distance,
                  max_age = max_age,
                  n_init = n_init,
                  max_cosine_distance = max_dist,
                  nn_budget = nn_budget,
                    embedder = embedder,
                    embedder_model_name = embedder_model_name,
                    embedder_wts = embedder_wts)



  def loadModelStrongSort(self):
    loadConfig = config.yaml_load()
    fp16 = loadConfig.get("strong_sort")['fp16']
    device = loadConfig.get("strong_sort")['device']
    max_iou_distance = loadConfig.get("strong_sort")['max_iou_distance']
    max_age = loadConfig.get("strong_sort")['max_age']
    n_init = loadConfig.get("strong_sort")['n_init']
    max_dist = loadConfig.get("strong_sort")['max_dist']
    nn_budget = loadConfig.get("strong_sort")["nn_budget"]
    model_weights = loadConfig.get("strong_sort")["embedder_wts"]
    self.MAX_AGE = max_age
    self.INIT_MAX_AGE = n_init
    return StrongSORT(model_weights = model_weights,
                  device = device,
                  fp16 = fp16,
                  max_iou_distance = max_iou_distance,
                  max_age = max_age,
                  n_init = n_init,
                  max_dist = max_dist,
                  nn_budget = nn_budget)


  def loadModelByteTracker(self):
    loadConfig = config.yaml_load().get("byte_track")
    parser = argparse.ArgumentParser("Tracker args!")
    parser.add_argument("--fuse_score",dest="fuse_score",default=loadConfig['fuse_score'],action="store_true",help="Fuse conv and bn for testing.",)
    parser.add_argument("--track_high_thresh", type=float, default=loadConfig['track_high_thresh'], help="track high thresh")
    parser.add_argument("--track_low_thresh", type=float, default=loadConfig['track_low_thresh'], help="track_low_thresh")
    parser.add_argument("--new_track_thresh", type=float, default=loadConfig['new_track_thresh'], help="track_low_thresh")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=loadConfig['track_buffer'], help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=loadConfig['match_thresh'], help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    self.MAX_AGE = loadConfig['track_buffer']
    self.INIT_MAX_AGE = loadConfig['n_init']
    self.MATCH_THRESHOLD = loadConfig['match_thresh']
    return BYTETracker(parser.parse_args())


  def update_tracking(self, results, frame):
    detections = None
    if self.type_model == "DeepSort":
      detections = self.transformationDataDeepSort(results, frame)
    if self.type_model == "StrongSort":
      detections = self.transformationDataStrongSort(results, frame)
    if self.type_model == "ByteTracker":
      detections = self.transformationDataByteTracker(results, frame)
    return self.trackingDataObject(detections = detections)


  """
  This method using to transformation data input from each Detector to standard data to Tracking

  Parameters
  ----------
  results: Tensors
    Consists many data from Detector Yolo like: bbox, conf, class
  Img: String
    Origin path image

  Return
  ----------
  Arrays have form is: [boundingbox, conf, class, img, frame]
  """
  def transformationDataInputTracking(self, results, img):
    if self.type_model == "DeepSort":
      return self.transformationDataDeepSort(results, img)
    if self.type_model == "StrongSort":
      return self.transformationDataStrongSort(results, img)
    if self.type_model == "ByteTracker":
      return self.transformationDataByteTracker(results, img)
    return None
    #
    #
  """
  This method using to transformation to data input for DeepSort
  Parameters
  ----------
  results: Tensors
    Consists many data from Detector Yolo like: bbox, conf, class
  Img: String
    Origin path image
  
  Return
  ----------
  Arrays have form is: [raw_detections(xywh, conf, cls), frame]
  array [xyxy, xywh, conf, cls, extraction, id, max_age]
  """
  def transformationDataDeepSort(self, results, frame):
    if self.run_original:
      detection = []
      for result in results:
        for i in result.boxes:
          xyxy = self.to_xyxy(i)
          conf = self.to_conf(i)
          cls = self.to_cls(i)
          detection.append([xyxy, conf, cls])
      return {
        "detections": detection,
        "frame": frame
      }
    else:
      return results

  "results.append([xyxy, conf, cls])"
  def transformationDataStrongSort(self, results, frame):
    if self.run_original:
      detection = []
      for result in results:
        for i in result.boxes:
          xyxy = self.to_xyxy(i)
          conf = self.to_conf(i)
          cls = self.to_cls(i)
          detection.append([xyxy[0],xyxy[1],xyxy[2],xyxy[3], conf, cls])
      return {
        "detections": torch.tensor(detection),
        "frame": frame
      }
    else:
      return results

  def transformationDataByteTracker(self, results, frame):
    if self.run_original:
      return {
        "detections": results[0].boxes,
        "frame": frame
      }
    else:
      return results

  def trackingDataObject(self, detections):
    if self.run_original:
      if self.type_model == "DeepSort":
        return self.model.update_tracks(raw_detections = detections['detections'], frame = detections['frame'])
      if self.type_model == "StrongSort":
        return self.model.update(dets=detections['detections'], ori_img=detections['frame'])
      if self.type_model == "ByteTracker":
        return self.model.update(results=detections['detections'], img=detections['frame'])

    else:
      if self.type_model == "DeepSort":
        tracking =  self.model.update_tracks(raw_detections = detections['detections'], frame = detections['frame'],embeds = detections['embeds'])
        return self.transformResultsTrackingDeepSort(tracking)
      if self.type_model == "StrongSort":
        tracking = self.model.update(dets = detections['detections'], ori_img = detections['frame'], embeds = detections['embeds'])
        return self.transformResultsTrackingStrongSort(tracking)
      if self.type_model == "ByteTracker":
        tracking = self.model.update(results = detections['detections'], img = detections['frame'])
        return self.transformResultsTrackingByteTrack(tracking, detections['detections_ts'])
      return None

  def to_xywh(self,box):
    x = float(box.xywh.cpu().numpy()[0][0])
    y = float(box.xywh.cpu().numpy()[0][1])
    w = float(box.xywh.cpu().numpy()[0][2])
    h = float(box.xywh.cpu().numpy()[0][3])
    return [x,y,w,h]

  def to_xyxy(self,box):
    x1 = float(box.xyxy.cpu().numpy()[0][0])
    y1 = float(box.xyxy.cpu().numpy()[0][1])
    x2 = float(box.xyxy.cpu().numpy()[0][2])
    y2 = float(box.xyxy.cpu().numpy()[0][3])
    return [x1,y1,x2,y2]

  def to_conf(self,box):
    return box.conf.cpu().numpy()[0]

  def to_cls(self,box):
    return box.cls.cpu().numpy()[0]

  def filterTrackingDetections(self,detections):
    if self.type_model == "DeepSort":
      return self.filterTrackingDetectionsDeepSort(detections)
    if self.type_model == "StrongSort":
      return self.filterTrackingDetectionsStrongSort(detections)
    if self.type_model == "ByteTracker":
      return self.filterTrackingDetectionsByteTracker(detections)

  def filterTrackingDetectionsDeepSort(self,detections):
    store_matched_flags = [False] * len(self.DETECTIONS_STORES)
    store_matched_detections = [False] * len(detections['detections'])
    if len(self.DETECTIONS_STORES) == 0:
      return detections
    else:
      for index, embed in enumerate(detections['embeds']):
        embed_tensor = torch.tensor(np.array(embed)).unsqueeze(0)


        for idx, detection_store in enumerate(self.DETECTIONS_STORES):
          embed_store = torch.tensor(np.array(detection_store[2])).unsqueeze(0)

          if self.comparativeService.is_matched(embed_tensor, embed_store):
            self.DETECTIONS_STORES[idx][2] = embed
            self.DETECTIONS_STORES[idx][1] = self.INIT_MAX_AGE
            detections['detections'][index][4] = self.DETECTIONS_STORES[idx][0]
            store_matched_flags[idx] = True
            store_matched_detections[index] = True
            break

    is_tracking = False
    for idx, matched in enumerate(store_matched_detections):
      if not matched:
        is_tracking = True
        break
    for idx, matched in enumerate(store_matched_flags):
      if not matched:
        self.DETECTIONS_STORES[idx][1] += 1

    self.DETECTIONS_STORES = [
      store for store in self.DETECTIONS_STORES if store[1] < self.MAX_AGE
    ]
    return detections if is_tracking else []


  def filterTrackingDetectionsStrongSort(self,detections):
    if not self.DETECTIONS_STORES:
      return detections

    store_matched_flags = [False] * len(self.DETECTIONS_STORES)
    un_mask = torch.ones(detections['embeds'].size(0), dtype=torch.bool)

    for index, embed in enumerate(detections['embeds']):
      embed_tensor = torch.tensor(np.array(embed)).unsqueeze(0)
      for idx, store in enumerate(self.DETECTIONS_STORES):
        store_tensor = torch.tensor(np.array(store[2]), dtype=torch.float32).unsqueeze(0)
        if self.comparativeService.is_matched(embed_tensor,store_tensor):
          self.DETECTIONS_STORES[idx][2] = embed
          self.DETECTIONS_STORES[idx][1] = self.INIT_MAX_AGE
          detections['detections'][index][6] = self.DETECTIONS_STORES[idx][0]
          store_matched_flags[idx] = True
          un_mask[index] = True
          break

    is_tracking = False
    for idx, matched in enumerate(un_mask):
      if not matched:
        is_tracking = True
        break

    for idx, matched in enumerate(store_matched_flags):
      if not matched:
        self.DETECTIONS_STORES[idx][1] += 1

    self.DETECTIONS_STORES = [
      store for store in self.DETECTIONS_STORES if store[1] < self.MAX_AGE
    ]
    return detections if is_tracking else []




  #detections['detections_ts'] =  Bounding box | conf | tracking_id | is_matched
  # self.DETECTIONS_STORES = tracking-id | track_buffer (=max_age) | bounding-box
  def filterTrackingDetectionsByteTracker(self,detections):
    if not self.DETECTIONS_STORES:
      return detections
    store_boxes = torch.stack([store[2] for store in self.DETECTIONS_STORES])  # shape [N_store, 4]
    store_matched_flags = [False] * len(self.DETECTIONS_STORES)
    store_matched_detections = [False] * len(detections['detections_ts'])

    for index, detection in enumerate(detections['detections_ts']):
      det_box = detection[0].unsqueeze(0)
      ious = ops.box_iou(det_box, store_boxes)[0]
      max_iou, max_idx = torch.max(ious, dim=0)

      if max_iou >= self.MATCH_THRESHOLD:
        self.DETECTIONS_STORES[max_idx][0] = detection[0]
        self.DETECTIONS_STORES[max_idx][1] = self.INIT_MAX_AGE
        detections['detections_ts'][index][2] = detection[0]
        detections['detections_ts'][index][3] = True

        store_matched_flags[max_idx] = True
        store_matched_detections[index] = True

    is_tracking = not all(store_matched_detections)

    for idx, matched in enumerate(store_matched_flags):
      if not matched:
        self.DETECTIONS_STORES[idx][1] += 1

    self.DETECTIONS_STORES = [
      store for store in self.DETECTIONS_STORES if store[1] < self.MAX_AGE
    ]

    return detections if is_tracking else []





  def updateFilterTracking(self, results_tracking_un_matched):
    if len(results_tracking_un_matched) != 0:
      if self.type_model == "DeepSort":
        for index, detection_un_matched in enumerate(results_tracking_un_matched):
          if detection_un_matched[0] is not None:
            detection = [detection_un_matched[0], self.INIT_MAX_AGE, detection_un_matched[1]]
            self.DETECTIONS_STORES.append(detection)

      if self.type_model == "StrongSort":
        for index, detection_un_matched in enumerate(results_tracking_un_matched):
          if detection_un_matched[0] is not None:
            detection = [detection_un_matched[0], self.INIT_MAX_AGE, detection_un_matched[1]]
            self.DETECTIONS_STORES.append(detection)

      if self.type_model == "ByteTracker":
        for index, detection_un_matched in enumerate(results_tracking_un_matched):
          if detection_un_matched[0] is not None:
            detection = [detection_un_matched[0] , self.INIT_MAX_AGE, detection_un_matched[1]]
            self.DETECTIONS_STORES.append(detection)


  def transformResultsTrackingDeepSort(self, results_tracking):
    trackings = []
    for track in results_tracking:
      if (track.track_id is not None and
              track.is_confirmed() and
              track.age == self.INIT_MAX_AGE):
        trackings.append([track.track_id, track.features[0]])
    return trackings

  def transformResultsTrackingStrongSort(self, results_tracking):
    trackings = []
    for track in results_tracking:
      if track.age == self.INIT_MAX_AGE:
        trackings.append([track.track_id, track.features[0]])
    return trackings

  # self.DETECTIONS_STORES = tracking-id | track_buffer (=max_age) | bounding-box
  # Bounding box | conf | tracking-id | is_matched
  # coords.tolist() + [self.track_id, self.score, self.cls, self.idx]
  def transformResultsTrackingByteTrack(self, resultsTracking, detections):
    trackings = []
    detections_news = [d for d in detections if not d[3]]
    detection_map = {round(d[1], 4): d[0] for d in detections_news}
    for result in resultsTracking:
      conf = round(float(result[5]), 4)
      if conf in detection_map:
        trackings.append([result[4], detection_map[conf]])
    return trackings

