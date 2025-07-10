import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import socket

#offset transmit
Pi_UDP_IP = "172.20.10.12"
OFFSET_PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load YOLOv7
device = torch.device('cuda')

#load .pt weights file
weights_path = r'E:\SWS-2025\pythonProject\yolov7.pt'

# Attempt to load the model
model = attempt_load(weights_path, map_location=device)
model.to(device).eval()

# GStreamer pipeline to receive H264 via UDP
gst_pipeline = (
    "udpsrc port=5000 caps=\"application/x-rtp, media=video, encoding-name=H264, payload=96\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! appsink"
)


# Open the video stream using GStreamer
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture(0)  #local camera

# Check if the video stream is opened successfully
if not cap.isOpened():
    raise RuntimeError("unable to open the video stream")

print("detecting......")

# read frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    #the central position of the frame:(frame_center_x,frame_center_y) 
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2


    #pre_proccess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the YOLO's input size
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()
    img_tensor /= 255.0

    # Ensure the image tensor is on the correct device
    with torch.no_grad():
        pred = model(img_tensor)[0]  # 取 tuple 第0个元素

    # Apply non-max suppression, reserve the most confident detection
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Process the detections
    if pred is not None and len(pred):
        # Rescale boxes from img_size to frame size
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()

        # Draw bounding boxes and labels on the frame
        for *xyxy, conf, cls in pred:

            # label the class and confidence
            label = f'{int(cls)} {conf:.2f}'

            # Draw the bounding box
            plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

            #the central position of the target:(target_center_x,target_center_y)
            x1, y1, x2, y2 = map(int, xyxy)
            target_center_x = (x1+x2)//2
            target_center_y = (y1+y2)//2

            # Calculate offsets and dimensions
            offset = target_center_x - frame_center_x

            # Calculate the width and height of the bounding box
            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)

            # Draw the center point of the target
            msg = f"{offset}, {box_width}, {box_height}"
            sock.sendto(msg.encode(), (Pi_UDP_IP, OFFSET_PORT) )


    cv2.imshow('YOLOv7 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
