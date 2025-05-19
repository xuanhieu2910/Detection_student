import sys
import os
import load_config as config

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


class TrackingService:


  """
  array [xyxy, xywh, conf, cls, extraction, id, max_age]
  """
  DETECTIONS_STORES = []
  MAX_AGE = 25
  INIT_MAX_AGE = 1


  def __init__(self, typeModelTracking):
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
    if self.type_model == "BotSort":
      return self.loadModelBotSort()
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
    model_weights = "C:\\Users\\hieux\\Desktop\\Personal\\Master\\PROJECT\\tracker\\model_weight\\osnet_x0_25.pt"
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
    return BYTETracker(parser.parse_args())


  def loadModelBotSort(self):
    loadConfig = config.yaml_load().get("bot_sort")
    parser = argparse.ArgumentParser("Tracker args!")
    parser.add_argument("--fuse_score", dest="fuse_score", default=loadConfig['fuse_score'], action="store_true",
                        help="Fuse conv and bn for testing.", )
    parser.add_argument("--track_high_thresh", type=float, default=loadConfig['track_high_thresh'],
                        help="track high thresh")
    parser.add_argument("--track_low_thresh", type=float, default=loadConfig['track_low_thresh'], help="track_low_thresh")
    parser.add_argument("--new_track_thresh", type=float, default=loadConfig['new_track_thresh'], help="track_low_thresh")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=loadConfig['track_buffer'],
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=loadConfig['match_thresh'],
                        help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--appearance_thresh', type=float, default=loadConfig['appearance_thresh'],
                        help='appearance thresh')
    parser.add_argument('--proximity_thresh', type=float, default=loadConfig['proximity_thresh'], help='proximity thresh')
    parser.add_argument('--with_reid', type=float, default=loadConfig['with_reid'], help='with reid')
    parser.add_argument('--gmc_method', type=str, default=loadConfig['gmc_method'], help='gmc_method')
    return BOTSORT(parser.parse_args())


  def update(self, results, img):
    detections = None
    if self.type_model == "DeepSort":
      detections = self.transformationDataDeepSort(results, img)
    if self.type_model == "StrongSort":
      detections = self.transformationDataStrongSort(results, img)
    if self.type_model == "ByteTracker":
      detections = self.transformationDataByteTracker(results, img)
    if self.type_model == "BotSort":
      detections = self.transformationDataBotSort(results, img)
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
    if self.type_model == "BotSort":
      return self.transformationDataBotSort(results, img)
    return None

  
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
  def transformationDataDeepSort(self, results, img):
    detection = []
    frame = cv2.imread(img)
    for result in results:
        detection.append([result[0], result[2], result[3], result[4]])
    return {
      "detections": detection,
      "frame": frame
    }

  def transformationDataDeepSortRoot(self, results, img):
    detection = []
    frame = cv2.imread(img)
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


  def transformationDataStrongSort(self, results, img):
    detection = []
    frame = cv2.imread(img)
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

  def transformationDataByteTracker(self, results, img):
    return {
      "detections": results[0].boxes,
      "img": img
    }

  def transformationDataBotSort(self, results, img):
    detection = []
    for result in results:
      for i in result.boxes:
        xyxy = self.to_xyxy(i)
        conf = self.to_conf(i)
        cls = self.to_cls(i)
        detection.append([xyxy, conf, cls])
    return {
      "detections": detection,
      "img": img
    }

  def trackingDataObject(self, detections):
    # detections['detections'] = self.filterDetections(detections['detections'])
    if self.type_model == "DeepSort":
      tracking =  self.model.update_tracks(raw_detections = detections['detections'], frame = detections['frame'])
      return self.transformTrackingDeepSort(tracking)
    if self.type_model == "StrongSort":
      return self.model.update(dets = detections['detections'], ori_img = detections['frame'])
    if self.type_model == "ByteTracker":
      return self.model.update(results = detections['detections'], img = detections['img'])
    if self.type_model == "BotSort":
      for detection in detections['detections']:
        dataInitTrack = self.model.init_track(dets = detection[0],
                                        scores = [detection[1]],
                                        cls = [detection[2]],
                                        img = detections['img'])
        self.model.multi_predict(dataInitTrack)
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
    detections_un_matched = []
    detections_max_age = []
    if len(self.DETECTIONS_STORES) < 1:
      return {
        "detections_un_matched": detections,
        "detections_max_age": detections_max_age
      }
    else:
      for detectionStore in self.DETECTIONS_STORES:
        if detectionStore[6] >= self.MAX_AGE:
          detectionStore[6] = int(self.INIT_MAX_AGE)
          detections_max_age.append(detectionStore)
        else:
          detectionStore[6] += 1




    for detection in detections:
      matched = False
      for detectionStore in self.DETECTIONS_STORES:
        if self.comparativeService.is_matched(torch.tensor(detection[4]), detectionStore[4]):
          detection[5] = detectionStore[5]
          matched = True
          break
      if not matched:
        detections_un_matched.append(detection)
    return {
      "detections_un_matched": detections_un_matched,
      "detections_max_age": detections_max_age
    }


  def updateFilterTracking(self, detections, trackings):
    for index,detection in enumerate(detections):
        if trackings[index][2]:
          detection[5] = int(trackings[index][1])
          detection[6] = self.INIT_MAX_AGE
          self.DETECTIONS_STORES.append(detection)


  def transformTrackingDeepSort(self, resultsTracking):
    trackings = []
    for results in resultsTracking:
      trackings.append([results.to_tlwh(),results.track_id, results.is_confirmed(), results.age, results.features])
    return self.toSort(trackings)

  def toSort(self, trackings):
      return sorted(trackings, key=lambda x: x[0][0], reverse=False)


  def trackingDataObjectRoot(self, detections):
    # detections['detections'] = self.filterDetections(detections['detections'])
    if self.type_model == "DeepSort":
      return self.model.update_tracks(raw_detections = detections['detections'], frame = detections['frame'])
    if self.type_model == "StrongSort":
      return self.model.update(dets = detections['detections'], ori_img = detections['frame'])
    if self.type_model == "ByteTracker":
      return self.model.update(results = detections['detections'], img = detections['img'])
    if self.type_model == "BotSort":
      for detection in detections['detections']:
        dataInitTrack = self.model.init_track(dets = detection[0],
                                        scores = [detection[1]],
                                        cls = [detection[2]],
                                        img = detections['img'])
        self.model.multi_predict(dataInitTrack)
    return None



  def filterTrackingDetectionsNew(self):
    trackers = self.model.tracker.tracks
    data = []
    for tracker in trackers:
      data.append([tracker.track_id, tracker.is_confirmed(),tracker.age, tracker.to_tlwh(), tracker.features])
    return data
    # print(f"Trackers: {self.model.tracker.tracks}")
    # if len(self.model.tracker.tracks) < 1:
    #   return detections