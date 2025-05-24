import os
import cv2
import BodyPoseService as bps
import DetectorService as ds
import FacialService as fs
import TrackingService as ts
# import ComparetiveService as cs
import torch
import time


def loadDataset():
    pathRootDataset = "dataset\\Test2"
    imgs = []
    directionsData = os.listdir(pathRootDataset)
    for item in directionsData:
        imgs.append(os.path.join(pathRootDataset, item))
    return imgs


def main(run_original = True):
    typeTracking = ["DeepSort", "StrongSort", "ByteTracker", "BotSort"]
    """"
      BotSort tam thoi dung lai, can phai xem lai BotSort
    """
    type_model_tracking = typeTracking[1]
    # initialize
    detectionModel = ds.DetectorService(os.path.join(os.getcwd(),"yolov5nu.pt"), type_model_tracking)
    run_original = run_original
    trackingModel = ts.TrackingService(type_model_tracking, run_original)

    # facialModel = fs.FacialService()
    # bodyPoseModel = bps.BodyPoseService()

    imgs = loadDataset()
    time_start = time.time()

    for img in imgs:
        results = detectionModel.predict(img)

        if run_original:
            result_tracking = trackingModel.trackingDataObject(trackingModel.transformationDataInputTracking(results, img))
        else:
            # time_start_1 = time.time()
            detectionsTransform = detectionModel.transformResults(results, cv2.imread(img))
            # print(f"Time detection transformation : {time.time() - time_start_1}")
            # detectionsUnique = detectionModel.removeDuplicate(detectionsTransform)
            # time_start_2 = time.time()
            detectionsFilter = trackingModel.filterTrackingDetections(detectionsTransform)
            # print(f"Time detections filter : {time.time() - time_start_2}")
            if (len(detectionsFilter["detections_max_age"]) > 0):
                trackingModel.update_tracking(detectionsFilter["detections_max_age"], img)
            if (len(detectionsFilter["detections_un_matched"]) > 0):
                # time_start_3 = time.time()
                result_tracking_un_matched = trackingModel.update_tracking(detectionsFilter["detections_un_matched"], img)
                # print(f"Time detections un_matched : {time.time() - time_start_3}")
                # time_start_4 = time.time()
                trackingModel.updateFilterTracking(detectionsFilter["detections_un_matched"], result_tracking_un_matched)
                # print(f"Time update detections filter un_matched : {time.time() - time_start_4}")


        # resultFacial = facialModel.extractionFacial(img = img)
        # resultPoseBody = bodyPoseModel.extractionBodyPose(img = img)
    time_end = time.time()
    print("Time elapsed: " + str(time_end - time_start))

if __name__ == "__main__":
    main(run_original = True)