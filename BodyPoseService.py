import pandas as pd

import body_pose_extraction as bodyPoseExtraction


class BodyPoseService:

  def __init__(self):
    self.model = bodyPoseExtraction.loadModelPose()
  


  """
  Method to extraction body data from detections 

  Parameters
  ----------
  img: is path file input

  Return
  ------
  Frame Data related Body pose
  """
  def extractionBodyPose(self, img):
    dataBody = bodyPoseExtraction.handleOpenPosePytorch(tp = self.model, image = img)
    if type(dataBody) == list:
      dataBody = pd.DataFrame(dataBody)
    return dataBody