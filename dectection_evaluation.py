import torch
import sys
import os
sys.path.append(os.path.abspath("/content/drive/MyDrive/Research/Computer_Vision/StudentEngagement_2/detector_tracker"))
from ultralytics.utils import LOGGER
import pandas as pd
import os
from thoppip.thop import profile


def calculate_gflops(model, imgsz=(640, 640)):
    """Calculate GFLOPS for the model."""
    try:
        dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1]).to(model.device)
        flops, _ = profile(model.model, inputs=(dummy_input,))
        gflops = flops / 1e9  # Convert to GFLOPS
        return gflops
    except Exception as e:
        LOGGER.warning(f"Could not calculate GFLOPS: {e}")
        return None

def calculate_params(model):
    """Calculate total number of parameters in the model."""
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        return total_params
    except Exception as e:
        LOGGER.warning(f"Could not calculate parameters: {e}")
        return None

def calculate_attributes_metrics(metrics):
  return {
      "image": "-",
      "conf": sum(float(m['conf']) for m in metrics) / len(metrics),
      "preprocess_time_ms": sum( float(m['preprocess_time_ms']) for m in metrics) / len(metrics),
      "inference_time_ms": sum( float(m['inference_time_ms']) for m in metrics) / len(metrics),
      "postprocess_time_ms" : sum( float(m['postprocess_time_ms']) for m in metrics) / len(metrics),
      "total_time_ms" : sum( float(m['total_time_ms']) for m in metrics) / len(metrics),
      "average_time_ms": sum( float(m['average_time_ms']) for m in metrics) / len(metrics),
      "number_of_people_dec": sum( float(m['number_of_people_dec']) for m in metrics) / len(metrics),
      "process_time_tracker_ms": sum( float(m['process_time_tracker_ms']) for m in metrics) / len(metrics),
      "params": sum( float(m['params']) for m in metrics) / len(metrics),
      "gflops": sum( float(m['gflops']) for m in metrics) / len(metrics),
    }


def save_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


def save_metrics_facial_to_csv(extractions, filename):
    extractionOr = extractions[0]
    index = 1
    while index < len(extractions):
      extractionOr = pd.concat([extractionOr,extractions[index]], ignore_index=True)
      index += 1
    extractionOr.to_csv(filename)
    print(f"Metrics saved to {filename}")