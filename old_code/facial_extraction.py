import os
from feat.detector import Detector
import cv2
import pandas as pd
import load_config as config
import dectection_evaluation as deval
from old_code import body_pose_extraction as bodyPoseExtraction
import handle_label as handleLabel

pathRootDataset = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/dataset/CSSBD/Test"
pathFileWriteData = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/runs/facial/Test_03_05_2025.csv"


def loadFrames(images):
  frames = []
  for image in images:
    frames.append(cv2.imread(image))
  return frames


def loadDataset():
  imgs = []
  directionsData = os.listdir(pathRootDataset)
  for item in directionsData:
      imgs.append(os.path.join(pathRootDataset,item))
  print(imgs)
  return imgs

def loadModel():
  loadConfig = config.yaml_load().get("feat-fy")
  faceModel = loadConfig['face_model']
  landmarkModel = loadConfig['landmark_model']
  auModel = loadConfig['au_model']
  facePoseModel = loadConfig['facepose_model']
  emotionModel = loadConfig['emotion_model']
  identityModel = loadConfig['identity_model']
  detectorFacial = Detector(
    face_model=faceModel,
    landmark_model=landmarkModel,
    au_model=auModel,
    emotion_model=emotionModel,
    facepose_model=facePoseModel,
    identity_model = identityModel
  )
  return detectorFacial


def extraction_facial(model,image):
  data =  model.detect_image(input_file_list = image)
  dataAu = data.aus
  dataPose = data.poses
  dataFacial = pd.concat([dataAu,dataPose], axis = 1)
  dataFacial['cls'] = handleLabel.handleAssignNameClass(cls = os.path.basename(img).split(".")[0].split("_")[2])
  dataFacial['file_name'] = os.path.basename(img)
  return dataFacial


def concatBodyAndFacial(dataBody, dataFacial):
  concatBodyAndFacial = pd.concat([dataBody, dataFacial], axis=1)
  return concatBodyAndFacial



model = loadModel()
dataSet = loadDataset()
frames = loadFrames(dataSet)
modelPose = bodyPoseExtraction.loadModelPose()
#------------------------------------- ORIGINAL ------------------------------------------------------------------
index = 0
extractions = []
print(f"Total images: {len(dataSet)}")
for img in dataSet:
  print(f"Handle iamge: {index} - {img}")
  dataFacial = extraction_facial(model = model, image = img)
  dataBody = bodyPoseExtraction.handleOpenPosePytorch(tp = modelPose, image = img)
  if type(dataBody) == list:
    dataBody = pd.DataFrame(dataBody)
  if type(dataFacial) == list:
    dataFacial = pd.DataFrame(dataFacial)
  if type(dataBody) != list and type(dataFacial) != list:
    extractions.append(concatBodyAndFacial(dataBody, dataFacial))
  index += 1

print(f"Complete extraction feature facial!")
deval.save_metrics_facial_to_csv(extractions, filename = pathFileWriteData)
#------------------------------------- ORIGINAL ------------------------------------------------------------------



