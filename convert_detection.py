import numpy as np

def to_xywh(box):
  x = float(box.xywh.cpu().numpy()[0][0])
  y = float(box.xywh.cpu().numpy()[0][1])
  w = float(box.xywh.cpu().numpy()[0][2])
  h = float(box.xywh.cpu().numpy()[0][3])
  return [x,y,w,h]


def to_xyxy(box):
  x1 = int(box.xyxy.cpu().numpy()[0][0])
  y1 = int(box.xyxy.cpu().numpy()[0][1])
  x2 = int(box.xyxy.cpu().numpy()[0][2])
  y2 = int(box.xyxy.cpu().numpy()[0][3])
  return [x1,y1,x2,y2]

def to_conf(box):
  return float(box.conf.cpu().numpy()[0])

def to_cls(box):
  return int(box.cls.cpu().numpy()[0])

def convertToRawDectections(results):
  detection = []
  for result in results:
    for i in result.boxes:
      xywh = to_xywh(i)
      conf = to_conf(i)
      cls = to_cls(i)
      detection.append([xywh, conf, cls])
  return detection



def convertToStrongSort(results):
  detection = []
  for result in results:
    for i in result.boxes:
      xyxy = to_xyxy(i)
      conf = to_conf(i)
      cls = to_cls(i)
      detection.append(xyxy + [conf] + [cls])
  return detection

