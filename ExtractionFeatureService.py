import sys
import os
sys.path.append(os.path.abspath("tracker\\deep_sort_real_time\\deep_sort_real_time\\embedder"))
from embedder_pytorch import TorchReID_Embedder
import load_config as config


class ExtractionFeatureService:

    def __init__(self):
        self.model = self.initExtractionFeatureModel()


    def initExtractionFeatureModel(self):
        loadConfig = config.yaml_load().get("feature-extraction")
        model_name = loadConfig['model_name']

        model_wts_path = "C:\\Users\\hieux\\Desktop\\Personal\\Master\\PROJECT\\osnet_x0_25.pth"
        bgr = True
        gpu = False

        return TorchReID_Embedder(model_name = model_name,
                                  model_wts_path = model_wts_path,
                                  bgr = bgr,
                                  gpu = gpu)

    def extraction_feature(self, np_images):
        return self.model.predict(np_images = np_images)