import os
import cv2
import BodyPoseService as bps
import DetectorService as ds
import FacialService as fs
import TrackingService as ts
import pandas as pd
import keras
import numpy as np
import csv
# import ComparetiveService as cs
import torch
import time

model_classification = keras.models.load_model("best_model_resnet.h5")

def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def loadDataset():
    pathRootDataset = "dataset\\Test2"
    imgs = []
    directionsData = os.listdir(pathRootDataset)
    for item in directionsData:
        imgs.append(os.path.join(pathRootDataset, item))
    return imgs

def classification(resultFacial,resultPoseBody, result_classification):
    if resultPoseBody is not None:
        features = pd.concat([resultPoseBody,resultFacial],axis = 1)
        #classification
        #print(features)
        result = model_classification.predict(features)
        count = (result >= 0.5).astype(int).flatten()
        result_classification.append(count[0])

def handleOriginal(detectionModel, trackingModel, facialModel, bodyPoseMode, results, frame):
    result_tracking = trackingModel.trackingDataObject(
        trackingModel.transformationDataInputTracking(results, frame))
    detection = trackingModel.transformationDataInputTracking(results, frame)['detections']
    return detection
    #resultFacial = facialModel.extractionFacial(img=img)
    #resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
    #classification(resultFacial,resultPoseBody)


def handleUpgrade(detectionModel, trackingModel, facialModel, bodyPoseMode, results, frame):
    detection = detectionModel.transformResults(results, frame)
    detectionsFilter = trackingModel.filterTrackingDetections(detection)
    if len(detectionsFilter) > 0:
        trackingModel.updateFilterTracking(trackingModel.update_tracking(detectionsFilter, frame))
    return detection['detections']
    #resultFacial = facialModel.extractionFacial(img=img)
    #resultPoseBody = bodyPoseModel.extractionBodyPose(img=img)
    #classification(resultFacial,resultPoseBody)

def handlePipelineNotSkipDetection(isOriginal, type_model_tracking, detectionModel, trackingModel, facialModel, bodyPoseMode, img, total_time_tracking, total_time_classification, result_classification):
    start_track = time.time()
    frame = cv2.imread(img)
    results = detectionModel.predict(img)
    if isOriginal:
        detection = handleOriginal(detectionModel=detectionModel,
                       trackingModel=trackingModel,
                       facialModel=facialModel,
                       bodyPoseMode=bodyPoseMode,
                       results=results,
                       frame=frame)
    else:
        detection = handleUpgrade(detectionModel=detectionModel,
                      trackingModel=trackingModel,
                      facialModel=facialModel,
                      bodyPoseMode=bodyPoseMode,
                      results=results,
                      frame=frame)
    total_time_tracking += (time.time() - start_track)
    if type_model_tracking == 0:
        list_detection = [detect[0] for detect in detection]
    elif type_model_tracking == 1:
        all_detection = detection.tolist()
        list_detection = []
        for sublist in all_detection:
            list_detection.append(sublist[:4])
    elif type_model_tracking ==2:
        list_detection = detection.xyxy.tolist()

    #resultFacial = facialModel.extractionFacial(frame)
    #resultPoseBody = bodyPoseMode.extractionBodyPose(frame, list_detection)
    start_classification = time.time()
    #classification(resultFacial,resultPoseBody,result_classification)
    total_time_classification += (time.time() - start_classification)
    return total_time_tracking, total_time_classification
def handlePipelineSkipDetection(detectionModel, trackingModel, facialModel, bodyPoseMode, img, total_time_tracking, total_time_classification, result_classification):
    start_track = time.time()
    frame = cv2.imread(img)
    dataTracking = trackingModel.DETECTIONS_STORES
    detect = dataTracking[-1]
    total_time_tracking += (time.time() - start_track)
    #detection = [tlwh_to_xyxy(detect[3])]
    #resultFacial = facialModel.extractionFacial(frame)
    #resultPoseBody = bodyPoseMode.extractionBodyPose(frame, detection)
    start_classification = time.time()
    #classification(resultFacial,resultPoseBody,result_classification)
    total_time_classification += (time.time() - start_classification)
    return total_time_tracking, total_time_classification

def main(run_original = True):
    typeTracking = ["DeepSort", "StrongSort", "ByteTracker"]
    type_model_tracking = typeTracking[2]

    detectionModel = ds.DetectorService(os.path.join(os.getcwd(),"yolo11n.pt"), type_model_tracking)
    run_original = run_original
    trackingModel = ts.TrackingService(type_model_tracking, run_original)
    facialModel = fs.FacialService()
    bodyPoseModel = bps.BodyPoseService()

    imgs = loadDataset()
    loop_test = 1
    frame_count = 0
    detection_interval = 1
    total_time_tracking = 0
    total_time_classification = 0
    result_classification = []


    for i in range(loop_test):
        for img in imgs:
            if len(trackingModel.DETECTIONS_STORES) == 0:
                total_time_tracking, total_time_classification = handlePipelineNotSkipDetection(isOriginal=run_original,
                               detectionModel=detectionModel,
                               type_model_tracking = 2,
                               trackingModel=trackingModel,
                               facialModel=facialModel,
                               bodyPoseMode=bodyPoseModel,
                               img=img,
                               total_time_tracking = total_time_tracking,
                               total_time_classification = total_time_classification,
                               result_classification = result_classification)
            else:
                if (frame_count % detection_interval) == 0:
                    total_time_tracking, total_time_classification = handlePipelineNotSkipDetection(isOriginal = run_original,
                                   detectionModel = detectionModel,
                                   type_model_tracking =2,
                                   trackingModel = trackingModel,
                                   facialModel = facialModel,
                                   bodyPoseMode = bodyPoseModel,
                                   img = img,
                                   total_time_tracking = total_time_tracking,
                                   total_time_classification = total_time_classification,
                                   result_classification = result_classification)
                    frame_count = 0
                else:
                    total_time_tracking,  total_time_classification = handlePipelineSkipDetection(detectionModel = detectionModel,
                                                trackingModel=trackingModel,
                                                facialModel=facialModel,
                                                bodyPoseMode = bodyPoseModel,
                                                img=img,
                                                total_time_tracking = total_time_tracking,
                                                total_time_classification = total_time_classification,
                                                result_classification = result_classification)
                frame_count += 1
        if i == 0:        
            with open('output.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for value in result_classification:
                    writer.writerow([value])
                result_classification = []
        
    print("Average classification time is {}".format(total_time_classification/loop_test/len(imgs)))
    print("Total time tracking is {}".format(total_time_tracking/loop_test/len(imgs)))
if __name__ == "__main__":
    main(run_original = True)
