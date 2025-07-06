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


def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def ltwh_to_xyxy(ltwh):
    x, y, w, h = ltwh
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def loadDataset():
    pathRootDataset = "dataset\\Test"
    imgs = []
    directionsData = os.listdir(pathRootDataset)
    for item in directionsData:
        imgs.append(os.path.join(pathRootDataset, item))
    return imgs

def handle_classification(resultFacial, resultPoseBody, result_classification, model_classification):
    if resultPoseBody is not None:
        features = pd.concat([resultPoseBody,resultFacial],axis = 1)
        #classification
        #print(features)
        result = model_classification.predict(features)
        count = (result >= 0.5).astype(int).flatten()
        result_classification.append(count[0])


def handleUpgrade(detectionModel, trackingModel, results, frame):
    detection = detectionModel.transformResults(results, frame)
    detectionsFilter = trackingModel.filterTrackingDetections(detection)
    if len(detectionsFilter) > 0:
        trackingModel.updateFilterTracking(trackingModel.update_tracking(detectionsFilter, frame))
    return detection['detections']



def handleNotSkipDetection(type_model_tracking,detectionModel,trackingModel,
                            facialModel,bodyPoseMode,img,
                            total_time_tracking,total_time_classification,result_classification):
    start_track = time.time()
    frame = cv2.imread(img)
    results = detectionModel.predict(img)
    detection = handleUpgrade(detectionModel=detectionModel,
                      trackingModel=trackingModel,
                      results=results,
                      frame=frame)

    total_time_tracking += (time.time() - start_track)

    if type_model_tracking == "DeepSort":
        list_detection = [detect[0] for detect in detection]
    elif type_model_tracking == "StrongSort":
        all_detection = detection.tolist()
        list_detection = []
        for sublist in all_detection:
            list_detection.append(sublist[:4])
    elif type_model_tracking == "ByteTracker":
        list_detection = detection.xyxy.tolist()

    #resultFacial = facialModel. (frame)
    #resultPoseBody = bodyPoseMode.extractionBodyPose(frame, list_detection)
    # start_classification = time.time()
    # handle_classification(resultFacial,resultPoseBody,result_classification)
    # total_time_classification += (time.time() - start_classification)
    return total_time_tracking, total_time_classification


def handlePipelineSkipDetection(detectionModel,
                                trackingModel,
                                facialModel,
                                bodyPoseMode,
                                img,
                                total_time_tracking,
                                total_time_classification,
                                result_classification):
    start_track = time.time()
    frame = cv2.imread(img)
    dataTracking = trackingModel.DETECTIONS_STORES
    detect = dataTracking[-1]
    total_time_tracking += (time.time() - start_track)
    #detection = [tlwh_to_xyxy(detect[3])]
    #resultFacial = facialModel.extractionFacial(frame)
    #resultPoseBody = bodyPoseMode.extractionBodyPose(frame, detection)
    start_classification = time.time()
    #handle_classification(resultFacial,resultPoseBody,result_classification)
    total_time_classification += (time.time() - start_classification)
    return total_time_tracking, total_time_classification

def transform_result_tracking_original(type_tracking, result_tracking):
    list_detection = []
    if type_tracking == "DeepSort":
        list_detection = [np.append(detect.to_tlwh(),detect.track_id) for detect in result_tracking]
    # elif type_tracking == "StrongSort":
    #     all_detection = result_tracking.tolist()
    #     list_detection = []
    #     for sublist in all_detection:
    #         list_detection.append(sublist[:4])
    # elif type_tracking == "ByteTracker":
    #     list_detection = result_tracking.xyxy.tolist()
    return list_detection



"""
---------------------------------------------------- CHẠY PHIÊN BẢN GỐC ------------------------------------------------
"""
def handle_detection_tracking_process_original(img, frame, detectionModel, trackingModel, type_tracking):
    start_detection_time = time.time()
    results = detectionModel.predict(img)
    time_detection = (time.time() - start_detection_time)

    start_tracking_time = time.time()
    tracking = run_tracking_original(trackingModel=trackingModel, results=results, frame=frame)
    time_tracking = (time.time() - start_tracking_time)
    list_detection = transform_result_tracking_original(type_tracking=type_tracking,
                                                        result_tracking=tracking)

    return time_detection, time_tracking, list_detection


def handle_facial_body_process_original(frame, result_detection_tracking, facialModel, bodyPoseModel,type_tracking):
    result_facial_bodypose_list = []
    for detect in result_detection_tracking:
        if type_tracking == "DeepSort":
            x1, y1, x2, y2, id = map(float, detect[:5])
            person = frame[int(y1):int(y2), int(x1):int(x2)]
            resultFacial = facialModel.extractionFacial(person)
            resultPoseBody = bodyPoseModel.extractionBodyPose(person)
            combined = pd.concat([resultPoseBody, resultFacial], axis=1)
            combined["ID"] = int(id)
            result_facial_bodypose_list.append(combined)
        elif type_tracking == "StrongSort":
            pass
        elif type_tracking == "ByteTracker":
            pass
    return pd.concat(result_facial_bodypose_list, ignore_index=True)


def run_end_to_end_original(run_original, version_yolo, type_tracking, classification_model):

    # Loading các model
    detectionModel = ds.DetectorService(modelWeight = os.path.join(os.getcwd(), version_yolo),
                                        type_model_tracking = type_tracking)
    trackingModel = ts.TrackingService(typeModelTracking = type_tracking,
                                       run_original = run_original)
    model_classification = keras.models.load_model(classification_model)
    facialModel = fs.FacialService()
    bodyPoseModel = bps.BodyPoseService()

    # Loading dataset
    imgs = loadDataset()

    loop_executed = 1

    total_time_detection = 0
    total_time_tracking = 0
    total_time_facial = 0

    total_time_classification = 0


    result_classification = []

    for i in range(loop_executed):
        for img in imgs:
            start_track = time.time()
            frame = cv2.imread(img)
            result_detection_tracking = handle_detection_tracking_process_original(img,frame, detectionModel, trackingModel,type_tracking)
            result_facial_bodypose = handle_facial_body_process_original(frame, result_detection_tracking[2], facialModel, bodyPoseModel, type_tracking)

            print(result_facial_bodypose)
            # start_classification = time.time()
            # handle_classification(resultFacial,resultPoseBody,result_classification)
            # total_time_classification += (time.time() - start_classification)
        # print("Average classification time is {}".format(total_time_classification/loop_executed/len(imgs)))
        print("Total time detection is {}".format(total_time_detection / loop_executed / len(imgs)))
        print("Total time tracking is {}".format(total_time_tracking/loop_executed/len(imgs)))
        print("Total time facial is {}".format(total_time_facial/loop_executed/len(imgs)))



def run_tracking_original(trackingModel, results, frame):
    return trackingModel.trackingDataObject(
        trackingModel.transformationDataInputTracking(results, frame))

"""
------------------------------------------ CHẠY BẢN UPGRADE ----------------------------------------------------------
"""
def run_end_to_end_upgrade(run_original, version_yolo, type_tracking, classification_model):
    run_original = run_original


    #Loading các model
    detectionModel = ds.DetectorService(os.path.join(os.getcwd(),version_yolo),
                                        type_tracking)
    model_classification = keras.models.load_model(classification_model)
    trackingModel = ts.TrackingService(type_tracking, run_original)
    facialModel = fs.FacialService()
    bodyPoseModel = bps.BodyPoseService()


    #Loading dataset
    imgs = loadDataset()


    loop_executed = 1
    frame_count = 0
    detection_interval = 1
    total_time_tracking = 0
    total_time_classification = 0
    result_classification = []


    for i in range(loop_executed):
        for img in imgs:
            if len(trackingModel.DETECTIONS_STORES) == 0:
                (total_time_tracking,
                 total_time_classification) = handleNotSkipDetection(detectionModel=detectionModel,
                               type_model_tracking = type_tracking,
                               trackingModel=trackingModel,
                               facialModel=facialModel,
                               bodyPoseMode=bodyPoseModel,
                               img=img,
                               total_time_tracking = total_time_tracking,
                               total_time_classification = total_time_classification,
                               result_classification = result_classification)

            else:
                if (frame_count % detection_interval) == 0:
                    (total_time_tracking,
                     total_time_classification) = handleNotSkipDetection(
                                   detectionModel = detectionModel,
                                   type_model_tracking = type_tracking,
                                   trackingModel = trackingModel,
                                   facialModel = facialModel,
                                   bodyPoseMode = bodyPoseModel,
                                   img = img,
                                   total_time_tracking = total_time_tracking,
                                   total_time_classification = total_time_classification,
                                   result_classification = result_classification)
                    frame_count = 0
                else:
                    (total_time_tracking,
                     total_time_classification) = handlePipelineSkipDetection(detectionModel = detectionModel,
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

    print("Average classification time is {}".format(total_time_classification/loop_executed/len(imgs)))
    print("Total time tracking is {}".format(total_time_tracking/loop_executed/len(imgs)))

def main(run_original = True,
         version_yolo = "yolov5nu.pt",
         type_tracking = "DeepSort",
         classification_model = "best_model_dnn.h5"):
    if run_original:
        run_end_to_end_original(run_original, version_yolo, type_tracking, classification_model)
    else:
        run_end_to_end_upgrade(run_original, version_yolo, type_tracking, classification_model)



if __name__ == "__main__":
    """    
    Selected tracking: "DeepSort" | "StrongSort" | "ByteTracker"
    
    Selected version yolo:  yolov5nu.pt | yolov7n.pt | yolov8n.pt | yolo11n.pt 
    
    Selected classification model: "best_model_dnn.h5" | "best_model_resnet.h5"
    """
    main(run_original = True,
         version_yolo = "yolov5nu.pt",
         type_tracking = "DeepSort",
         classification_model = "best_model_dnn.h5")