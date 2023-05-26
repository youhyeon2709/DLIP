import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ground truth 파일과 detect 파일의 경로
ground_truth_file = 'groundtruth.txt'
detect_file = 'counting_result.txt'

# 파일에서 데이터 읽기
with open(ground_truth_file, 'r') as file:
    ground_truth = file.readlines()

with open(detect_file, 'r') as file:
    detected = file.readlines()

# 데이터 정리
ground_truth = [int(value.strip().split(',')[1]) for value in ground_truth]
detected = [int(value.strip().split()[1]) for value in detected]

# 공통된 데이터 샘플만 사용
min_samples = min(len(ground_truth), len(detected))
ground_truth = ground_truth[:min_samples]
detected = detected[:min_samples]

# Accuracy 계산
accuracy = np.mean(np.array(ground_truth) == np.array(detected))

# 결과 출력
print("Accuracy:", accuracy)
