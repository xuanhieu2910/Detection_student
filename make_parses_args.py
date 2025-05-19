import argparse
import os
import os.path as osp
import time
import cv2
import torch



def make_parser():
    parser = argparse.ArgumentParser("Tracker args!")
    parser.add_argument("--fuse_score",dest="fuse_score",default=False,action="store_true",help="Fuse conv and bn for testing.",)
    parser.add_argument("--track_high_thresh", type=float, default=0.25, help="track high thresh")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="track_low_thresh")
    parser.add_argument("--new_track_thresh", type=float, default=0.25, help="track_low_thresh")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    return parser.parse_args()