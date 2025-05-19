import sys
import os
sys.path.append(os.path.abspath('torchreid\\torchreid'))
from reid.metrics import distance
import load_config as config

class ComparetiveService:
    def __init__(self):
        self.metric = config.yaml_load().get("comparative")["metric"]
        self.distance_match = config.yaml_load().get("comparative")["distance_match"]


    def is_matched(self, input1, input2):
        dis = distance.compute_distance_matrix(input1 = input1,
                                       input2 = input2,
                                       metric = self.metric)
        print("DISTANCE: ", dis)
        return dis <= self.distance_match