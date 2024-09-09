import torch
from yolov5.detect import run
 
# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
 
result = run(weights='weights/best.pt', source=0)

