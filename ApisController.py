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
    typeTracking = ["DeepSort", "StrongSort", "ByteTracker"]

    type_model_tracking = typeTracking[0]
    # initialize
    detectionModel = ds.DetectorService(os.path.join(os.getcwd(),"yolov5nu.pt"), type_model_tracking)
    run_original = run_original
    trackingModel = ts.TrackingService(type_model_tracking, run_original)

    # facialModel = fs.FacialService()
    # bodyPoseModel = bps.BodyPoseService()

    imgs = loadDataset()
    loop_test = 1
    total_time = 0
    for i in range(loop_test):
        for img in imgs:
            frame = cv2.imread(img)
            results = detectionModel.predict(img)
            start_track = time.time()
            # result_tracking = trackingModel.trackingDataObject(trackingModel.transformationDataInputTracking(results, frame))
            #------------------------------------------------------------------------------------------------------
            detectionsTransform = detectionModel.transformResults(results, frame)
            print(f"Detection transform: {detectionsTransform['detections']}")
            detectionsFilter = trackingModel.filterTrackingDetections(detectionsTransform)
            if len(detectionsFilter) > 0:
                    result_tracking_un_matched = trackingModel.update_tracking(detectionsFilter,frame)
                    trackingModel.updateFilterTracking(result_tracking_un_matched)
            # #
            total_time += (time.time() - start_track)

            # resultFacial = facialModel.extractionFacial(img = img)
            # resultPoseBody = bodyPoseModel.extractionBodyPose(img = img)

    print("Total time average is {}".format(total_time/loop_test))
if __name__ == "__main__":
    # detectionsUnique = detectionModel.removeDuplicate(detectionsTransform)
    main(run_original = False)