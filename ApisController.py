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


def main():
    typeTracking = ["DeepSort", "StrongSort", "ByteTracker", "BotSort"]
    """"
      BotSort tam thoi dung lai, can phai xem lai BotSort
    """
    # initialize
    detectionModel = ds.DetectorService("C:\\Users\\hieux\\Desktop\\Personal\\Master\\PROJECT\\yolov5nu.pt")
    trackingModel = ts.TrackingService("ByteTracker")

    # facialModel = fs.FacialService()
    # bodyPoseModel = bps.BodyPoseService()

    # modelCompare = cs.ComparetiveService()


    imgs = loadDataset()
    time_start = time.time()

    for img in imgs:
        results = detectionModel.predict(img)
        # print(type(results))

        # trackings = trackingModel.trackingDataObjectRoot(trackingModel.transformationDataByteTracker(results, img))
        # for tracking in trackings:
        #     print(f"x1: {tracking[0]} - y1: {tracking[1]} - x2: {tracking[2]} - y2: {tracking[3]} - ID: {tracking[4]} - Conf: {tracking[5]}")
        detectionsTransform = detectionModel.transformResultsByteTracker(results, cv2.imread(img))
        # detectionsUnique = detectionModel.removeDuplicate(detectionsTransform)
        # time_start_a = time.time()
        detectionsFilter = trackingModel.filterTrackingDetectionsByteTracker(detectionsTransform)
        if (len(detectionsFilter["detections_max_age"]) > 0):
            trackingModel.update(detectionsFilter["detections_max_age"], img)
        if (len(detectionsFilter["detections_un_matched"]) > 0):
            trackings = trackingModel.update(detectionsFilter["detections_un_matched"][2], img)
            trackingModel.updateFilterTracking(detectionsFilter["detections_un_matched"], trackings)

        # resultFacial = facialModel.extractionFacial(img = img)
        # resultPoseBody = bodyPoseModel.extractionBodyPose(img = img)
    time_end = time.time()
    print("Time elapsed: " + str(time_end - time_start))

if __name__ == "__main__":
    main()