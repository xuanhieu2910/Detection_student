from tensorflow import keras
import sys
import os
model = keras.models.load_model(os.path.join(sys.path[0],"best_model_resnet.h5"))