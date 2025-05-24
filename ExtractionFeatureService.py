import sys
import os
sys.path.append(os.path.abspath("tracker\\deep_sort_real_time\\deep_sort_real_time\\embedder"))
from embedder_pytorch import TorchReID_Embedder
sys.path.append(os.path.abspath("tracker\\strongsort\\strongsort"))
from reid_multibackend import ReIDDetectMultiBackend
import load_config as config

class ExtractionFeatureService:

    def __init__(self, type_model):
        self.type_model = type_model
        self.model = self.initExtractionFeatureModel()

    def initExtractionFeatureModel(self):
        if self.type_model == "DeepSort":
            return self.loadModelExtractionFeatureModelDeepSort()
        if self.type_model == "StrongSort":
            return self.loadModelExtractionFeatureModelStrongSort()
        if self.type_model == "ByteTracker":
            return self.loadModelExtractionFeatureModelByteTracker()
        if self.type_model == "BotSort":
            return self.loadModelExtractionFeatureModelBotSort()

    def extractFeatures(self, np_images):
        # if self.type_model == "DeepSort":
        #     return self.extractionFeatureDeepSort(np_images)
        # if self.type_model == "StrongSort":
            return self.extractionFeatureStrongSort(np_images)
        # if self.type_model == "ByteTracker":
        #     return self.extractionFeatureByteTracker(np_images)
        # if self.type_model == "BotSort":
        #     return self.extractionFeatureBotSort(np_images)

    def loadModelExtractionFeatureModelDeepSort(self):
        loadConfig = config.yaml_load().get("feature-extraction")
        model_name = loadConfig['model_name']
        model_wts_path = loadConfig['model_wts_path']
        bgr = True
        gpu = False
        return TorchReID_Embedder(model_name = model_name,
                                  model_wts_path = model_wts_path,
                                  bgr = bgr,
                                  gpu = gpu)

    def loadModelExtractionFeatureModelStrongSort(self):
        model_weights = config.yaml_load().get("feature-extraction")['model_weight']
        fp16 = config.yaml_load().get("strong_sort")['fp16']
        device = config.yaml_load().get("strong_sort")['device']
        return ReIDDetectMultiBackend(weights=model_weights,
                                      device=device,
                                      fp16=fp16)


    def loadModelExtractionFeatureModelByteTracker(self):
        pass

    def loadModelExtractionFeatureModelBotSort(self):
        pass


    def extractionFeatureDeepSort(self, np_images):
        return self.model.predict(np_images = np_images)

    def extractionFeatureStrongSort(self, np_images):
        return self.model(np_images)

    def extractionFeatureByteTracker(self, np_images):
        pass

    def extractionFeatureBotSort(self, np_images):
        pass