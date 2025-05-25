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
    i = 0
    aver_time = 0
    print("i = {}".format(i))
    for i in range(5):
        time_start = time.time()
        for img in imgs:
            results = detectionModel.predict(img)

            if run_original:
                result_tracking = trackingModel.trackingDataObject(
                    trackingModel.transformationDataInputTracking(results, img))
            else:
                # time_str = time.time()
                detectionsTransform = detectionModel.transformResults(results, cv2.imread(img))
                # print("Detection time: ", time.time() - time_str)
                # detectionsUnique = detectionModel.removeDuplicate(detectionsTransform)
                # time_file = time.time()
                detectionsFilter = trackingModel.filterTrackingDetections(detectionsTransform)

                # print("Time filter: ", time.time() - time_file)
                # print("Detection filter: ")
                # for de in detectionsTransform['detections']:
                #     print(de)
                if len(detectionsFilter['detections']) > 0:
                    # time_tracking = time.time()
                    result_tracking_un_matched = trackingModel.update_tracking(detectionsFilter,img)
                    trackingModel.updateFilterTracking(detectionsFilter, result_tracking_un_matched)
                    # print("Time tracking: ", time.time() - time_tracking)
                # resultFacial = facialModel.extractionFacial(img = img)
                # resultPoseBody = bodyPoseModel.extractionBodyPose(img = img)
        time_end = time.time()
        aver_time += time_end - time_start
    print("Total time average is {}".format(aver_time/5))

if __name__ == "__main__":
    main(run_original = True)