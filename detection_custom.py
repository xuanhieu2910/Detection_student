import sys
import os
sys.path.append(os.path.abspath("/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/detector_tracker"))
from ultralytics import YOLO
import torch
import cv2
import pandas as pd
import time
import dectection_evaluation as deval
import load_config as config
import numpy as np
import tracker_custom as tracker_custom
import convert_detection as convertDetection
import time
# from tracker.deepsort

# # Initial properties
# pathRootDataset = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/dataset/CSSBD/Test_2"
# fileSaveResult = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/runs/detection/28042025/yolov5_s_byte_track_runs_dection_Sub_8_combine_cpu.csv"
# fileSaveCrop = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/runs/data_crop/Sub_8_combine"
# classPerson = 0

# def loadFrames(images):
#   frames = []
#   for image in images:
#     frames.append(cv2.imread(image))
#   return frames

# # Load model detection
# def loadModel(nameModelWeight):
#   model = YOLO(nameModelWeight)
#   return model

# # Load dataset
# def loadDataset():
#   imgs = []
#   directionsData = os.listdir(pathRootDataset)
#   for item in directionsData:
#       imgs.append(os.path.join(pathRootDataset,item))
#   return imgs


# # Format result detection
# def formatResultDetection(pathImg,results,params,gflops, timeTracker):
#   for result in results: 
#     totalTime  = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
#     return {
#       "image": os.path.basename(pathImg),
#       "conf":float(np.float32(np.mean(result.boxes.conf.cpu().numpy()))),
#       "preprocess_time_ms": result.speed['preprocess'],
#       "inference_time_ms": result.speed['inference'],
#       "postprocess_time_ms" : result.speed['postprocess'],
#       "total_time_ms" : totalTime,
#       "average_time_ms":float(totalTime/3),
#       "number_of_people_dec":len(result.boxes.conf.cpu().numpy()),
#       "process_time_tracker_ms":timeTracker,
#       "params":params,
#       "gflops":gflops
#     }


# # Crop detection
# def cropData(x1, y1, x2, y2, frame, pathImg):
#   cropped_img = frame[y1 : y2, x1 : x2]
#   cv2.imwrite(pathImg, cropped_img)


# def predict(img):
#   results = modelDetection.predict(img,
#                           imgsz = imgsz,
#                           iou = iou,
#                           conf = conf,
#                           classes = classes)

# # Runining model
# #Loading model
# dataset = loadDataset()
# modelDetection = loadModel()
# modelTracker = tracker_custom.loadModel()
# frames = loadFrames(dataset)
# #Init attributes
# params = deval.calculate_params(modelDetection)
# gflops = deval.calculate_gflops(modelDetection)
# loadConfig = config.yaml_load().get("yolo")
# imgsz = loadConfig['imgsz']
# iou = loadConfig['iou']
# conf = loadConfig['conf']
# maxDet = loadConfig['max_det']
# classes = loadConfig["classes"]
# metricsResult = []
# abc = []
# indexImage = 0
# startProcess = time.time()

# "yolov5m.pt"

# for img in dataset:
#   results = modelDetection.predict(img,
#                           imgsz = imgsz,
#                           iou = iou,
#                           conf = conf,
#                           classes = classes)
  