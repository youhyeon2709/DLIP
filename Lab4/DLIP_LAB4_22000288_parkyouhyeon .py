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

# 비디오 재생 속도 설정
fast_forward_factor = 500  # 재생 속도를 2배로 설정

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




# 텍스트 파일 경로
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

            # 프레임을 빠르게 재생하기 위해 일부 프레임 건너뛰기
            #for _ in range(fast_forward_factor - 1):
            #    cap.read()

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