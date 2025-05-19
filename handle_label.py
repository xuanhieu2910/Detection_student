import os
import csv
import shutil

pathRoot = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement/runs/data_crop/Sub_8_combine"
pathRootAssignLabel = "/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement/runs/label/results/Sub_8_combine"

def getFileDataOrignalImages():
    return os.listdir(pathRoot)


def handleLabel(img):
    baseNamePath = os.path.basename(img).split["."][0]
    label = handleAssignNameClass(baseNamePath.split("_")[2])
    pathImage = os.path.join(pathRootAssignLabel ,baseNamePath + "_" + label + ".png")
    return pathImage

def copyFile(originalFile, destinationFile):
  shutil.copyfile(src = originalFile, dst = destinationFile)


def handleAssignNameClass(cls):
    if cls == "1": return "1"
    elif cls == "2": return "1"
    elif cls == "3": return "1"
    elif cls == "4": return "0"
    elif cls == "5": return "0"
    elif cls == "6": return "0"
    elif cls == "7": return "0"


# dataImages = getFileDataOrignalImages()


# for dataImage in dataImages:
#   destinationFile = handleLabel(dataImage)
#   copyFile(originalFile = dataImage, destinationFile = destinationFile)