#CPU
import os
import time
import math
import cv2
import numpy as np
import pandas as pd
from openpose_pytorch import torch_openpose, util
from google.colab.patches import cv2_imshow

tp = torch_openpose.torch_openpose('body_25')
num_bf = 70

input_base_path = './train_latest'
output_base_path = './traincsvopenpose'
input_folders = os.listdir(input_base_path)

os.makedirs(output_base_path, exist_ok=True)

# Đếm tổng thời gian từng phần
time_pose_total = 0
time_distance_total = 0
time_angle_total = 0
image_count = 0

start_all = time.time()
print("logging has started")

for folder in input_folders:
    folder_path = os.path.join(input_base_path, folder)
    train_file_path = os.path.join(output_base_path, folder)
    os.makedirs(train_file_path, exist_ok=True)

    images_list = os.listdir(folder_path)
    print("Processing folder:", folder, "| Total images:", len(images_list))

    for images in images_list:
        image_count += 1
        img_path = os.path.join(folder_path, images)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2_imshow(img)
        poses = tp(img)
        img_in = util.draw_bodypose(img, poses,'body_25')
        cv2_imshow(img_in)

        image_width, image_height = img.shape[1], img.shape[0]
        input_features = np.ones((1, num_bf)) * (1e-6)
        iter_features = 0

        # ⏱ Đo thời gian keypoint
        start_pose = time.time()
        poses = tp(img)
        time_pose_total += time.time() - start_pose

        if poses:
            # Lấy 10 điểm quan trọng
            for i in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16]:
                if len(poses[0]) > i and poses[0][i][2] > 0.1:
                    input_features[0][iter_features] = poses[0][i][0] / image_width
                    iter_features += 1
                    input_features[0][iter_features] = poses[0][i][1] / image_height
                    iter_features += 1
                else:
                    input_features[0][iter_features] = 1e-6
                    iter_features += 1
                    input_features[0][iter_features] = 1e-6
                    iter_features += 1

            # ⏱ Đo thời gian tính khoảng cách
            start_dist = time.time()
            for i in range(10):
                for j in range(i + 1, 10):
                    dx = input_features[0][2 * i] - input_features[0][2 * j]
                    dy = input_features[0][2 * i + 1] - input_features[0][2 * j + 1]
                    distance = math.sqrt(dx**2 + dy**2)
                    input_features[0][iter_features] = distance
                    iter_features += 1
            time_distance_total += time.time() - start_dist

            # ⏱ Đo thời gian tính góc
            start_angle = time.time()
def calc(x1, y1, x2, y2):
  dx = x1 - x2
  dy = y1 - y2
  return math.atan2(dy, dx)

def safe_angle(a, ref):
  result = (a - ref) % (2 * math.pi)
  return result

            # Các góc
            angle_01 = calc(*input_features[0][0:4])
            input_features[0][iter_features] = safe_angle(angle_01, math.pi / 2); iter_features += 1

            angle_23 = calc(*input_features[0][4:8])
            input_features[0][iter_features] = safe_angle(math.pi - angle_23, 0); iter_features += 1

            angle_34 = calc(*input_features[0][6:10])
            input_features[0][iter_features] = safe_angle(angle_34, math.pi / 2); iter_features += 1

            angle_56 = calc(*input_features[0][10:14])
            input_features[0][iter_features] = safe_angle(angle_56, 0); iter_features += 1

            angle_67 = calc(*input_features[0][12:16])
            input_features[0][iter_features] = safe_angle(math.pi / 2 - angle_67, 0); iter_features += 1

            time_angle_total += time.time() - start_angle

        # Lưu ra .csv
        file_name = os.path.splitext(images)[0]
        df = pd.DataFrame(input_features)
        df.to_csv(os.path.join(train_file_path, f"{file_name}.csv"), index=False, header=False)

end_all = time.time()

# ✅ In kết quả thời gian trung bình
print("\n--- Tổng kết thời gian ---")
print(f"Ảnh xử lý: {image_count}")
print(f"Thời gian trung bình pose: {(time_pose_total / image_count) * 1000:.6f} ms")
print(f"Thời gian trung bình distance: {(time_distance_total / image_count) * 1000:.6f} ms")
print(f"Thời gian trung bình angle: {(time_angle_total / image_count) * 1000:.6f} ms")
print(f"Tổng thời gian chương trình: {(end_all - start_all):.6f} s")