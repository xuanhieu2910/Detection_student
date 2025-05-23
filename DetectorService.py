import sys
import os

sys.path.append(os.path.abspath("detector_tracker"))
from ultralytics import YOLO
import load_config as config
import ExtractionFeatureService as efs
import ComparetiveService as compares



class DetectorService:

  loadConfig = config.yaml_load().get("yolo")
  imgsz = loadConfig['imgsz']
  iou = loadConfig['iou']
  conf = loadConfig['conf']
  maxDet = loadConfig['max_det']
  classes = loadConfig["classes"]

  def __init__(self, modelWeight):
    self.model = YOLO(modelWeight)
    self.modelExtraction = efs.ExtractionFeatureService()
    self.modelComparative = compares.ComparetiveService()

  def predict(self, img):
    result =  self.model.predict(img,
                          imgsz = self.imgsz,
                          iou = self.iou,
                          conf = self.conf,
                          classes = self.classes)

    return result

  """
    array [xyxy, xywh, conf, cls, extraction, id, max_age]
  """

  def transformResults(self, detections, frame):
      results = []
      crops = []
      xyxy_list = []


      for detection in detections:
          for box in detection.boxes:
              xyxy = self.to_xyxy(box)
              xywh = self.to_xywh(box)
              conf = self.to_conf(box)
              cls = self.to_cls(box)

              # Lưu thông tin crop để xử lý batch sau
              crop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
              crops.append(crop)
              xyxy_list.append((xyxy, xywh, conf, cls))

      # Chạy batch extraction một lần duy nhất
      features = self.modelExtraction.extraction_feature(np_images=crops)

      # Gộp kết quả
      for (xyxy, xywh, conf, cls), feat in zip(xyxy_list, features):
          results.append([xyxy, xywh, conf, cls, feat, 0, 0])

      return self.toSort(results)

  def transformResultsByteTracker(self, detections, frame):
      results = []
      crops = []
      xyxy_list = []

      for detection in detections:
          for box in detection.boxes:
              box = box
              xyxy = self.to_xyxy(box)
              xywh = self.to_xywh(box)
              # Lưu thông tin crop để xử lý batch sau
              crop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
              crops.append(crop)
              xyxy_list.append((xyxy, xywh, box))

      # Chạy batch extraction một lần duy nhất
      features = self.modelExtraction.extraction_feature(np_images=crops)

      # Gộp kết quả
      for (xyxy, xywh, box), feat in zip(xyxy_list, features):
          results.append([xyxy, xywh, box, feat, 0, 0])

      return self.toSort(results)



  def transformResultsRoot(self, detections, frame):
      results = []
      for detection in detections:
          for i in detection.boxes:
            xyxy = self.to_xyxy(i)
            conf = self.to_conf(i)
            cls = self.to_cls(i)
            results.append([xyxy,conf,cls])
      return self.toSort(results)

  def to_xywh(self, box):
      x = float(box.xywh.cpu().numpy()[0][0])
      y = float(box.xywh.cpu().numpy()[0][1])
      w = float(box.xywh.cpu().numpy()[0][2])
      h = float(box.xywh.cpu().numpy()[0][3])
      return [x, y, w, h]

  def to_xyxy(self, box):
      x1 = float(box.xyxy.cpu().numpy()[0][0])
      y1 = float(box.xyxy.cpu().numpy()[0][1])
      x2 = float(box.xyxy.cpu().numpy()[0][2])
      y2 = float(box.xyxy.cpu().numpy()[0][3])
      return [x1, y1, x2, y2]

  def to_conf(self, box):
      return box.conf.cpu().numpy()[0]

  def to_cls(self, box):
      return box.cls.cpu().numpy()[0]

  def to_extraction(self, xyxy, frame):
      imgCrop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
      return self.modelExtraction.extraction_feature(np_images = imgCrop)

  def removeDuplicate(self, detections):
      res = []
      n = len(detections)
      for i in range(n):
          if i != 0 and self.modelComparative.is_matched(detections[i-1][4],detections[i][4]):
              "Compare confidence, get higher"
              res[len(res) - 1] = detections[i-1] if  detections[i][2] < detections[i-1][2] else detections[i]
          else:
              res.append(detections[i])
      return res

  def toSort(self, detections):
      return sorted(detections, key=lambda x: x[0][0], reverse=False)
