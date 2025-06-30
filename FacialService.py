from feat.detector import Detector
import cv2
import pandas as pd
import load_config as config


class FacialService:

  loadConfig = config.yaml_load().get("feat-fy")
  faceModel = loadConfig['face_model']
  landmarkModel = loadConfig['landmark_model']
  auModel = loadConfig['au_model']
  facePoseModel = loadConfig['facepose_model']
  emotionModel = loadConfig['emotion_model']
  identityModel = loadConfig['identity_model']


  def __init__(self):
    self.model = Detector(
    face_model = self.faceModel,
    landmark_model = self.landmarkModel,
    au_model = self.auModel,
    emotion_model = self.emotionModel,
    facepose_model= self.facePoseModel,
    identity_model = self.identityModel
    )


  """
  Method to extraction facial data from detections 

  Parameters
  ----------
  img: is path file input

  Return
  ------
  Frame Data related AUS and Head
  """
  def extractionFacial(self, frame):
    landmarks = self.detectLandmarks(frame)
    dataAu = self.detectAus(frame = frame, landmarks = landmarks)
    dataAu.rename(columns = {0:"AU01",1:"AU02",2:"AU03",3:"AU03",4:"AU04",5:"AU05",6:"AU06",7:"AU07"
      ,8:"AU08",9:"AU09",10:"AU10",11:"AU11",12:"AU12",13:"AU13",14:"AU14",15:"AU15",16:"AU16",17:"AU17",18:"AU18",19:"AU19"}, inplace = True)

    dataPose = self.detectFacePose(frame, landmarks)
    dataPose.rename(columns = {0:"Pitch", 1:"Roll", 2:"Yaw"}, inplace = True)

    dataFacial = pd.concat([dataAu,dataPose], axis = 1)
    # dataFacial['cls'] = handleLabel.handleAssignNameClass(cls = os.path.basename(img).split(".")[0].split("_")[2])
    # dataFacial['file_name'] = os.path.basename(img)
    if type(dataFacial) == list:
      dataFacial = pd.DataFrame(dataFacial)
    return dataFacial


  def detectFaces(self, frame):
    return self.model.detect_faces(frame)

  def detectLandmarks(self, frame):
    detected_faces = self.detectFaces(frame = frame)
    return self.model.detect_landmarks(frame = frame, detected_faces = detected_faces)

  def detectFacePose(self, frame, landmarks=None):
    data = self.model.detect_facepose(frame = frame, landmarks = landmarks)['poses']
    if type(data) == list:
      return pd.DataFrame(data.pop())
    return data

  def detectAus(self, frame, landmarks):
    data = self.model.detect_aus(frame = frame, landmarks = landmarks)
    if type(data) == list:
      return pd.DataFrame(data.pop())
    return data





