# DLIP_LAB4_22000288_parkyouhyeon

**Date : 2023.05.25**

**Author: YouHyeon-Park**

**Github:** [https://github.com/youhyeon2709/DLIP/tree/main/Lab4](https://github.com/youhyeon2709/DLIP/tree/main/Lab4) 

**Demo Video:** [https://youtu.be/3xTi4Vao8yU](https://youtu.be/3xTi4Vao8yU)

# **Introduction**

> **Goal** : Create a program that displaysthat (1) counts the number of vehicles in the parking lot and (2) display the number of available parking space.
> 

The pre-trained yolov5 model counts the parked vehicles within the frame and the available parking spaces. For the given dataset, the maximum available parking space is 13. If the current number of vehicles is more than 13, then, the available space should display as ‘0’.

# **Requirement**

****Hardware****

intel core i7

****Software Installation****

CUDA 10.2

cudatoolkit 10.2

Python 3.9.12

Pytorch 1.9.1

Torchvision 0.10.1 

Torchaudio 0.9.1

YOLO v5

# **Dataset**

Using the pretrained model YOLOv5x .pt file.

**Dataset link:** [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt) 

# **Procedure**

****Follow the steps for setting YOLO V5 in the Anaconda.****

```python
# create a conda env name=py39 (you can change your env name)
  conda create -n py39 python=3.9.12
  conda activate py39

#Install Numpy, OpenCV, Matplot, Jupyter
	conda activate py39
	conda install -c anaconda seaborn jupyter
	pip install opencv-python

# Install Pytorch (CPU Only)
	conda install -c anaconda seaborn jupyter
	conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch
	pip install opencv-python torchsummary

# clone yolov5
  git clone https://github.com/ultralytics/yolov5.git
```

****Detecting car using yolov5.****

> Perform preprocessing to detect vehicle objects in the frame, filter the results based on confidence, and extract class and bounding box information about the vehicle.
> 
- Code
    
    ```python
     def car_detection(frame):
        # preprocessing frame
        image = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
        results = model(image)
        results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] > 0.4]  # Only keep results with a confidence level of 0.4 or higher
    
        # Filtering for vehicle classes
        car_results = results.pandas().xyxy[0]  # class, bounding box from result
        car_results = car_results[car_results['name'] == 'car']  # Select vehicle class only
    
        return car_results
    ```
    
    Creates an image object by converting a frame in BGR format to RGB format. This is done to convert to a standardized image format used by some libraries and models.
    
    Input the converted image into the object detection model to get the results, which include information about the location and class of various objects.
    
    Filter the results, keeping only those with a confidence of 0.4 or higher. This eliminates uncertain detection results with low confidence.
    
    Convert the filtered results to Pandas dataframe format and store them in the car_results variable. From car_results, select only those results that correspond to the vehicle class with a class of 'car'. This eliminates other object classes and extracts information about vehicles only.
    
- Result Image
    
    ![Untitled](DLIP_LAB4_22000288_parkyouhyeon%20a47d3a7766ef4f56a5ba7eddbaed89db/Untitled.png)
    

**Counting the number of vehicles in the parking lot.**

> Draws a box on the frame for the vehicle, checks to see if the vehicle is within the specified region (roi_points), and counts the number of vehicles.
> 
- Code
    
    ```python
    #Initialize roi_porints
    roi_points =np.array([(width*0, height*0.4),(width*0, height*0.615),(width*1, height*0.6),(width*1, height*0.4)],dtype=np.int32)
    
    def car_counting(car_results):
    
        car_count = 0
    
        # Drawing boxes on vehicles
        for _, detection in car_results.iterrows():
            class_name = detection['name']
            left, top, right, bottom = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
            confidence = detection['confidence']
            cv2.rectangle(frame, (left, top), (right, bottom), lime_green, 2)
            label = f"{class_name } {confidence:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lime_green, 2)
            #print(label)
            # Calculate the vehicle's center coordinates
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
    
            # Exclude vehicles outside the zone
            if not cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False) < 0:
                car_count += 1
    
        return car_count
    ```
    
    To determine if a vehicle is within a specified region (roi_points), one uses the center of the vehicle's bounding box. If the car's center coordinates are within the specified region (roi_points), increment 'car_count' by 1. 
    
    The 'cv2.pointPolygonTest' function is used to check whether a point is within a polygon.
    
    ```python
    parking_space = 13-car_count
    ```
    
    The calculation of the number of available parking spaces is obtained by subtracting the number of parked cars from the total number of parking spaces (13).
    
- Result Image
    
    ![Untitled](DLIP_LAB4_22000288_parkyouhyeon%20a47d3a7766ef4f56a5ba7eddbaed89db/Untitled%201.png)
    

# **Results and Analysis**

- Result Image
    
    ![Untitled](DLIP_LAB4_22000288_parkyouhyeon%20a47d3a7766ef4f56a5ba7eddbaed89db/Untitled%202.png)
    
    An excerpt from the result video above. You can see that the number of cars currently recognized is 9, but the number of parked cars is 8, and the number of available parking spaces is 13-8 = 5, which is a good result.
    
- Result  Video URL : [https://youtu.be/3xTi4Vao8yU](https://youtu.be/3xTi4Vao8yU)
- Table showing results up to the 2500th frame.
    
    
    | Frame(fps) | The number of vehicles in the parking lot |
    | --- | --- |
    | 1-180 | 13 |
    | 181-500 | 12 |
    | 501-770 | 11 |
    | 771-1058 | 10 |
    | 1059-1336 | 11 |
    | 1337-1412 | 10 |
    | 1413-1615 | 9 |
    | 1616-2017 | 8 |
    | 2018-2193 | 7 |
    | 2194-2302 | 6 |
    | 2303-2448 | 5 |
    | 2449-2500 | 4 |
    
- Table of values given from ground truth
    
    
    | Frame(fps) | The number of vehicles in the parking lot |
    | --- | --- |
    | 0-178 | 13 |
    | 179-499 | 12 |
    | 500-770 | 11 |
    | 771-1058 | 10 |
    | 1059-1335 | 11 |
    | 1336-1412 | 10 |
    | 1413-1500 | 9 |
    
    In this lab, one counted the number of cars in the parking lot per frame and compared it to the ground truth value. The frame segment with 13 cars is about 1 fps different, from 1-180 (fps) in the program created in this lab and 0-178 in ground truth. The 12 car segment has a difference of 1 fps, the first 11 car segment is 1 fps, the 10 car segment is 2 fps, the second 1 car segment is 1 fps, the second 10 car segment is 10 fps, and the 9 car segment is 1 fps. Most of the bins showed a difference of 1 fps. This is expected to be due to the difference in deciding whether to count parked cars or not based on where the center of the detected car is in the ROI range. Aside from the difference in frame indexes caused by the difference in ROI coverage, the program one created in this lab is very similar to the results from ground truth. The objectives of this lab have been met.
    
    In this lab, one of the difficulties we encountered was duplicate recognition of one car and counting it as two. To solve this problem, we set the confidence threshold to 0.4, so that if the confidence is lower than 0.4, it is not detected at all. After applying the above method, the problem was solved immediately.
    
- Accuracy
    
    1500 fps 까지의 groundtruth와 프로그램의 결과값을 비교하여 Accuracy를 구했다. 
    
    ```python
    with open('groundtruth.txt', 'r') as file:
        ground_truth= file.readlines()
    
    with open('counting_result.txt', 'r') as file:
        detected = file.readlines()
    
    # Preprocess data.
    ground_truth = [int(value.strip().split(',')[1]) for value in ground_truth]
    detected = [int(value.strip().split()[1]) for value in detected]
    
    # Use only common data samples.
    min_samples = min(len(ground_truth), len(detected))
    ground_truth = ground_truth[:min_samples]
    detected = detected[:min_samples]
    
    # Accuracy
    accuracy = np.mean(np.array(ground_truth) == np.array(detected))
    
    # 결과 출력
    print("Accuracy:", accuracy)
    ```
    
    The **`strip()`** function removes spaces and newline characters before and after the string, and the **`split()`** function creates a list of elements separated by spaces. Here, we select only the second element (index 1) and convert it to an integer using the **`int()`** function.
    
    Truncate the length of the ground_truth and detected lists to fit the value of min_samples. This ensures that both lists are the same length, and that only matching data samples are used.
    
    Convert the ground_truth and detected lists to NumPy arrays to create a Boolean array representing the results of comparing elements in the same position. Compute the average of this Boolean array to get accuracy. This is a value that represents the percentage of correctly detected elements.
    
    **Accuracy has a value of about 0.996.**
    

# Conclusion

In this lab, one created a program that loads a video, detects cars, and counts the number of parked cars and the number of available parking spaces. One used the pre-trained YOLOv5x.pt file to detect cars. In order to prevent uncertain detections, one only recognized cars with a confidence of 0.4 or higher. One only recognized parked cars if the center of the box drawn on the detected car was within the ROI area of the parking space. The number of available parking spaces was calculated by subtracting the number of currently parked cars from the total number of parking spaces, which is 13. The results were analyzed and it was found that the accuracy was 99.6%, which is almost the same as the groundtruth result. The program of this lab was successfully implemented.

# Appendix

```python
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import time
import torch
from pathlib import Path
from PIL import Image

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='.\\yolov5x.pt', force_reload=True).to(device)

#open video
cap = cv2.VideoCapture('DLIP_parking_test_video.avi')
_, frame = cap.read()

#height, width, channels = frame.shape
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output AVI file settings
output_path = 'path_to_output_video.avi'  
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)

# Resetting the frame count
frame_count = 0

# Initialize the start time
start_time = 0

#Initialize roi_porints
roi_points =np.array([(width*0, height*0.4),(width*0, height*0.615),(width*1, height*0.6),(width*1, height*0.4)],dtype=np.int32)

#Initialize color
sky_blue = (255,191,0)
lime_green = (0,255,227)

def calculate_FPS(start_time, end_time):

    #The process of getting FPS
    processing_time = end_time - start_time
    fps = (1 / processing_time) 
    rounded_fps = round(fps, 2) 
    return  rounded_fps

def car_detection(frame):
    # preprocessing frame
    image = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
    results = model(image)
    results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] > 0.4]  # Only keep results with a confidence level of 0.4 or higher

    # Filtering for vehicle classes
    car_results = results.pandas().xyxy[0]  # class, bounding box from result
    car_results = car_results[car_results['name'] == 'car']  # Select vehicle class only

    return car_results

def car_counting(car_results):

    car_count = 0

    # Drawing boxes on vehicles
    for _, detection in car_results.iterrows():
        class_name = detection['name']
        left, top, right, bottom = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
        confidence = detection['confidence']
        cv2.rectangle(frame, (left, top), (right, bottom), lime_green, 2)
        label = f"{class_name } {confidence:.2f}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lime_green, 2)
        #print(label)
        # Calculate the vehicle's center coordinates
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Exclude vehicles outside the zone
        if not cv2.pointPolygonTest(np.array(roi_points), (center_x, center_y), False) < 0:
            car_count += 1

    return car_count

output_txt_path = 'output.txt'
try:
    with open(output_txt_path, 'w') as output_file:

        # If the file was successfully opened, call the write() method to write the contents.

        # Play a video
        while cap.isOpened():

            # Read the current frame
            ret, frame = cap.read()

            if not ret:
                break

            # Measure when the algorithm starts.
            start_time = time.time()

            # Increase frame count
            frame_count += 1

            # detecting car using pretrained model
            car_results = car_detection(frame)  

            # Count the number of vehicles and available parking spaces
            car_count = car_counting(car_results)
            parking_space = 13-car_count

            # Displaying counts in frames
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sky_blue, 2)
            cv2.putText(frame, f"the number of vehicles : {car_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sky_blue, 2)
            cv2.putText(frame, f"the number of available parking space : {parking_space}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sky_blue, 2)
            
            # Measure when the algorithm finishes.
            end_time = time.time()
            fps = calculate_FPS(start_time, end_time)
            cv2.putText(frame, f"FPS: {fps}" , (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sky_blue, 2)  

            # Write frame_count and car_count per frame to a text file
            output_file.write(f"{frame_count } {car_count}\n")  
            print(f"{frame_count: } {car_count}\n")      

            # Display frames
            cv2.imshow('Video', frame)

            # Adding frames to the output video
            output_video.write(frame)

            # Wait for keyboard input and stop playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except IOError:
    print("Failed to open the output file.")

# close file
output_file.close()

# Releasing resources
cap.release()
cv2.destroyAllWindows()
```