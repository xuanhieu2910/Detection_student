import os
import cv2
import BodyPoseService as bps
import DetectorService as ds
import FacialService as fs
import TrackingService as ts
import pandas as pd
import keras
import numpy as np
# import ComparetiveService as cs
import torch
import time

result_classification = [] 
model_classification = keras.models.load_model("best_model_resnet.h5")

def loadDataset():
    pathRootDataset = "dataset\\Test2"
    imgs = []
    directionsData = os.listdir(pathRootDataset)
    for item in directionsData:
        imgs.append(os.path.join(pathRootDataset, item))
    return imgs

def classification(resultFacial,resultPoseBody):
    if not resultPoseBody.empty:
        features = pd.concat([resultPoseBody,resultFacial],axis = 1)
        #classification
        #print(features)
        result = model_classification.predict(features)
        #if len(detectionsFilter) > 0:
        count = (result >= 0.5).astype(int).flatten()
        result_classification.append(count)
        #counts = np.bincount(count, minlength=2)
        #if i == 0:
        #    total_negative += counts[0]
        #    total_positive += counts[1]

def handleOriginal(detectionModel, trackingModel, facialModel, bodyPoseMode, results, frame):
    result_tracking = trackingModel.trackingDataObject(
        trackingModel.transformationDataInputTracking(results, frame))
    # resultFacial = facialModel.extractionFacial(img=img)
    # resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
    # classification(resultFacial,resultPoseBody)


def handleUpgrade(detectionModel, trackingModel, facialModel, bodyPoseMode, results, frame):
    detectionsFilter = trackingModel.filterTrackingDetections(detectionModel.transformResults(results, frame))
    if len(detectionsFilter) > 0:
        trackingModel.updateFilterTracking(trackingModel.update_tracking(detectionsFilter, frame))
    # resultFacial = facialModel.extractionFacial(img=img)
    # resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
    # classification(resultFacial,resultPoseBody)

def handlePipelineNotSkipDetection(isOriginal, detectionModel, trackingModel, facialModel, bodyPoseMode, img):
    frame = cv2.imread(img)
    results = detectionModel.predict(img)
    if isOriginal:
        handleOriginal(detectionModel=detectionModel,
                       trackingModel=trackingModel,
                       facialModel=None,
                       bodyPoseMode=None,
                       results=results,
                       frame=frame)
        # resultFacial = facialModel.extractionFacial(img=img)
        # resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
        # classification(resultFacial,resultPoseBody)
    else:
        handleUpgrade(detectionModel=detectionModel,
                      trackingModel=trackingModel,
                      facialModel=None,
                      bodyPoseMode=None,
                      results=results,
                      frame=frame)
        # resultFacial = facialModel.extractionFacial(img=img)
        # resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
        # classification(resultFacial,resultPoseBody)

def handlePipelineSkipDetection(detectionModel, trackingModel, facialModel, bodyPoseMode, img):
    frame = cv2.imread(img)
    dataTracking = trackingModel.DETECTIONS_STORES

def main(run_original = True):
    typeTracking = ["DeepSort", "StrongSort", "ByteTracker"]
    type_model_tracking = typeTracking[0]

    detectionModel = ds.DetectorService(os.path.join(os.getcwd(),"yolo11s.pt"), type_model_tracking)
    run_original = run_original
    trackingModel = ts.TrackingService(type_model_tracking, run_original)
    # facialModel = fs.FacialService()
    # bodyPoseModel = bps.BodyPoseService()

    imgs = loadDataset()
    loop_test = 1
    total_time = 0
    frame_count = 0
    detection_interval = 3

    for i in range(loop_test):
        for img in imgs:

            if len(trackingModel.DETECTIONS_STORES) == 0:
                handlePipelineNotSkipDetection(isOriginal=run_original,
                               detectionModel=detectionModel,
                               trackingModel=trackingModel,
                               facialModel=None,
                               bodyPoseMode=None,
                               img=img)
            else:
                if frame_count % detection_interval == 0:
                    start_track = time.time()
                    handlePipelineNotSkipDetection(isOriginal = run_original,
                                   detectionModel = detectionModel,
                                   trackingModel = trackingModel,
                                   facialModel = None,
                                   bodyPoseMode = None,
                                   img = img)
                    total_time += (time.time() - start_track)
                    frame_count = 0
                else:
                    handlePipelineSkipDetection(detectionModel = detectionModel,
                                                trackingModel=trackingModel,
                                                facialModel=None,
                                                bodyPoseMode = None,
                                                img=img)
                frame_count += 1

    print("Total time average is {}".format(total_time/loop_test))
if __name__ == "__main__":
    main(run_original = False)
