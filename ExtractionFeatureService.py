import sys
import os
sys.path.append(os.path.abspath("tracker\\deep_sort_real_time\\deep_sort_real_time\\embedder"))
from embedder_pytorch import TorchReID_Embedder
import load_config as config

from tracker.strongsort.strongsort.reid_multibackend import ReIDDetectMultiBackend

class ExtractionFeatureService:

    def __init__(self):
        self.model_deepsort = self.initExtractionFeatureModelDeepSort()
        self.model_strongsort = self.initExtractionFeatureModelStrongSort()

    def initExtractionFeatureModelDeepSort(self):
        loadConfig = config.yaml_load().get("feature-extraction")
        model_name = loadConfig['model_name']

        model_wts_path = "D:\\AnhThienLe\\Intern_CV\\Detection_student\\osnet_x0_25.pth"
        bgr = True
        gpu = False

        return TorchReID_Embedder(model_name = model_name,
                                  model_wts_path = model_wts_path,
                                  bgr = bgr,
                                  gpu = gpu)
    def initExtractionFeatureModelStrongSort(self):
        loadConfig = config.yaml_load()
        model_weights = loadConfig.get("strong_sort")["embedder_wts"]
        fp16 = loadConfig.get("strong_sort")['fp16']
        device = loadConfig.get("strong_sort")['device']
        return ReIDDetectMultiBackend(weights = model_weights,
                  device = device,
                  fp16 = fp16)
    def extraction_feature(self, np_images, type):
        if (type == "DeepSort"):
            return self.model_deepsort.predict(np_images = np_images)
        elif (type=="StrongSort"):
            return self.model_strongsort(np_images)    
        elif (type=="ByteTrack"):
            pass
