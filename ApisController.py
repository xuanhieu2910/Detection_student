import os
# import ComparetiveService as cs
import time

import cv2
import keras
import numpy as np
import pandas as pd

import BodyPoseService as bps
import DetectorService as ds
import FacialService as fs
import TrackingService as ts


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
    pathRootDataset = "dataset\\Test2"
    imgs = []
    directionsData = os.listdir(pathRootDataset)
    for item in directionsData:
        imgs.append(os.path.join(pathRootDataset, item))
    return imgs

def handle_classification(features_facial_body_pose, result_classification, model_classification):
        result = model_classification.predict(features_facial_body_pose)
        count = (result >= 0.5).astype(int).flatten()
        result_classification.append(count[0])



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
    start_extract = time.time()
    if len(result_detection_tracking) > 0:
        result_facial_bodypose_list = []
        if type_tracking == "DeepSort":
            for detect in result_detection_tracking:
                x1, y1, x2, y2, id = map(float, detect[:5])
                person = frame[int(y1):int(y2), int(x1):int(x2)]
                resultFacial = facialModel.extractionFacial(person)
                resultPoseBody = bodyPoseModel.extractionBodyPose(person)
                combined = pd.concat([resultPoseBody, resultFacial], axis=1)
                combined["ID"] = int(id)
                result_facial_bodypose_list.append(combined)
        elif type_tracking == "StrongSort":
            for detect in result_detection_tracking:
                x1, y1, x2, y2, id = detect[:5]
                person = frame[int(y1):int(y2), int(x1):int(x2)]
                resultFacial = facialModel.extractionFacial(person)
                resultPoseBody = bodyPoseModel.extractionBodyPose(person)
                combined = pd.concat([resultPoseBody, resultFacial], axis=1)
                combined["ID"] = int(id)
                result_facial_bodypose_list.append(combined)
        elif type_tracking == "ByteTracker":
            for detect in result_detection_tracking:
                x1, y1, x2, y2, id = detect[:5]
                person = frame[int(y1):int(y2), int(x1):int(x2)]
                resultFacial = facialModel.extractionFacial(person)
                resultPoseBody = bodyPoseModel.extractionBodyPose(person)
                combined = pd.concat([resultPoseBody, resultFacial], axis=1)
                combined["ID"] = int(id)
                result_facial_bodypose_list.append(combined)
        time_extract = time.time() - start_extract
        return time_extract, pd.concat(result_facial_bodypose_list, ignore_index=True)
    time_extract = time.time() - start_extract
    return time_extract, []

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
    total_time_extract = 0
    total_time_classification = 0
    result_classification = []
    person_processed = 0

    for i in range(loop_executed):
        for img in imgs:
            start_track = time.time()
            frame = cv2.imread(img)
            result_detection_tracking = handle_detection_tracking_process_original(img,frame, detectionModel, trackingModel,type_tracking)
            #time_extract, result_facial_body_pose = handle_facial_body_process_original(frame, result_detection_tracking[2], facialModel, bodyPoseModel, type_tracking)
            person_processed += len(result_detection_tracking[2])
            total_time_detection += result_detection_tracking[0]
            total_time_tracking += result_detection_tracking[1]
            #total_time_extract += time_extract
            # Xử lý tiếp classification
            # handle_classification(result_facial_body_pose, result_classification, model_classification)


        # print("Average classification time is {}".format(total_time_classification/loop_executed/len(imgs)))
    print("Total time detection is {}".format(total_time_detection / loop_executed / len(imgs)))
    print("Total time tracking is {}".format(total_time_tracking/loop_executed/len(imgs)))
    print("Average time extract feature per person is {}".format(total_time_extract/loop_executed/person_processed))

def transform_result_tracking_original(type_tracking, result_tracking):
    list_detection = []
    if type_tracking == "DeepSort":
        list_detection = [np.append(detect.to_tlwh(),detect.track_id) for detect in result_tracking]
    elif type_tracking == "StrongSort":
        if len(result_tracking) > 0:
            list_detection = [np.append(tlwh_to_xyxy(detect.to_tlwh()),detect.track_id) for detect in result_tracking]
    elif type_tracking == "ByteTracker":
        for detect in result_tracking:
            list_detection.append(detect[:5])
    return list_detection

def run_tracking_original(trackingModel, results, frame):
    return trackingModel.trackingDataObject(
        trackingModel.transformationDataInputTracking(results, frame))

"""
------------------------------------------ CHẠY BẢN UPGRADE ----------------------------------------------------------
"""


def handle_facial_body_process_upgrade(frame, DETECTIONS_STORES, facialModel, bodyPoseModel, type_tracking):
    start_extract = time.time()
    result_facial_bodypose_list = []
    if type_tracking == "DeepSort":
        for detect in DETECTIONS_STORES:
            x1, y1, x2, y2 = detect[3][:]
            id = detect[0]
            person = frame[int(y1):int(y2), int(x1):int(x2)]
            resultFacial = facialModel.extractionFacial(person)
            resultPoseBody = bodyPoseModel.extractionBodyPose(person)
            combined = pd.concat([resultPoseBody, resultFacial], axis=1)
            combined["ID"] = int(id)
            result_facial_bodypose_list.append(combined)
    elif type_tracking == "StrongSort":
        for detect in DETECTIONS_STORES:
            x1, y1, x2, y2 = tlwh_to_xyxy(detect[3][:])
            id = detect[0]
            person = frame[int(y1):int(y2), int(x1):int(x2)]
            resultFacial = facialModel.extractionFacial(person)
            resultPoseBody = bodyPoseModel.extractionBodyPose(person)
            combined = pd.concat([resultPoseBody, resultFacial], axis=1)
            combined["ID"] = int(id)
            result_facial_bodypose_list.append(combined)
    elif type_tracking == "ByteTracker":
        for detect in DETECTIONS_STORES:
            x1, y1, x2, y2 = detect[2][:]
            id = detect[0]
            person = frame[int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item())]
            resultFacial = facialModel.extractionFacial(person)
            resultPoseBody = bodyPoseModel.extractionBodyPose(person)
            combined = pd.concat([resultPoseBody, resultFacial], axis=1)
            combined["ID"] = int(id)
            result_facial_bodypose_list.append(combined)
    time_extract = time.time() - start_extract
    return time_extract, pd.concat(result_facial_bodypose_list, ignore_index=True)


def run_end_to_end_upgrade(run_original, version_yolo, type_tracking, classification_model, detection_interval):

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
    detection_interval = detection_interval
    frame_count = 0
    total_time_detection = 0
    total_time_tracking = 0
    total_time_extract = 0
    total_time_classification = 0
    person_processed = 0 
    result_classification = []


    for i in range(loop_executed):
        for img in imgs:
            frame = cv2.imread(img)
            if len(trackingModel.DETECTIONS_STORES) == 0:
                start_detection_time = time.time()
                results = detectionModel.predict(img)
                total_time_detection += (time.time() - start_detection_time)
                start_tracking_time = time.time()
                detection_tracking_process_upgrade(detectionModel=detectionModel, trackingModel=trackingModel,
                                                   results=results, frame=frame)
                total_time_tracking += (time.time() - start_tracking_time)
            else:
                if frame_count != 0 and (frame_count % detection_interval == 0):
                        print("Run lại nè!!!!")
                        start_detection_time = time.time()
                        results = detectionModel.predict(img)
                        total_time_detection += (time.time() - start_detection_time)
                        start_tracking_time = time.time()
                        detection_tracking_process_upgrade(detectionModel=detectionModel, trackingModel=trackingModel,
                                                           results=results, frame=frame)
                        total_time_tracking += (time.time() - start_tracking_time)
                        frame_count = 0
                        continue
                frame_count += 1

            if len(trackingModel.DETECTIONS_STORES) > 0:
                person_processed += len(trackingModel.DETECTIONS_STORES)
                #time_extract, data_facial_body_pose = handle_facial_body_process_upgrade(frame, trackingModel.DETECTIONS_STORES,
                #                                                          facialModel, bodyPoseModel,type_tracking)
                #total_time_extract += time_extract
                #Xử lý tiếp classification
                #handle_classification(data_facial_body_pose, result_classification, model_classification)

        print("Total time detection is {}".format(total_time_detection / loop_executed / len(imgs)))
        print("Total time tracking is {}".format(total_time_tracking/loop_executed/len(imgs)))
        print("Average time extract feature per person is {}".format(total_time_extract/loop_executed/person_processed))


def detection_tracking_process_upgrade(detectionModel, trackingModel, results, frame):
    detection = detectionModel.transformResults(results, frame)
    detectionsFilter = trackingModel.filterTrackingDetections(detection)
    if len(detectionsFilter) > 0:
        trackingModel.updateFilterTracking(trackingModel.update_tracking(detectionsFilter, frame))


"----------------------------------------------------------------------------------------------------------------------"

def main(run_original = True,
         version_yolo = "yolo11n.pt",
         type_tracking = "DeepSort",
         classification_model = "best_model_dnn.h5",
         detection_interval = 3):
    if run_original:
        run_end_to_end_original(run_original, version_yolo, type_tracking, classification_model)
    else:
        run_end_to_end_upgrade(run_original, version_yolo, type_tracking, classification_model,detection_interval)



if __name__ == "__main__":
    """    
    Selected tracking: "DeepSort" | "StrongSort" | "ByteTracker"
    
    Selected version yolo:  yolov5nu.pt | yolov7n.pt | yolov8n.pt | yolo11n.pt 
    
    Selected classification model: "best_model_dnn.h5" | "best_model_resnet.h5"
    """
    main(run_original = False,
         version_yolo = "yolov8n.pt",
         type_tracking = "ByteTracker",
         classification_model = "best_model_dnn.h5",
         detection_interval = 3)