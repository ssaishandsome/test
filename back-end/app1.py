import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
from processor.AIDetector_pytorch import Detector

import torch
from flask_cors import CORS
import cv2

from yolov5.detect import run 
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.models.common import DetectMultiBackend






UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app1 = Flask(__name__,template_folder='F:\\2024-smallterm\Yolov5-Flask-VUE-master\Yolov5-Flask-VUE-master\\back-end\\templates')
app1.secret_key = 'secret!'
app1.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app1.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

CORS(app1, resources={r"/*": {"origins": "*"}})  # 配置CORS允许所有来源



# 添加header解决跨域
@app1.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


device = select_device('cpu')
model = DetectMultiBackend('F:\Yolov5-Flask-VUE-master\\back-end\weights\\best.pt', device=device, dnn=False, data=None)

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    #print("摄像头已打开")
    while True:
        ret, frame = cap.read()
        #print("读取到帧")

        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #print("帧已转换为 RGB 格式")
            
            # 调用 detect 函数，传入临时图片的路径
        annotated_frame = run(img=frame,model=model,device='cpu')
        print("目标检测完成")
            # 在图像上绘制检测结果
        for item in annotated_frame:
            print(item["coordinates"])
            try:
                # 检查coordinates是否包含正好4个元素
                if len(item["coordinates"]) == 4:
                    cx, cy, w, h = item["coordinates"]
                    conf = item["confidence"]
                    cls = item["class_id"]
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    # 在图像上绘制矩形和文本
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls}: {conf:.2f}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"检测到{cls}，置信度为{conf:.2f}")

                else:
                    print("跳过错误的元素：coordinates长度不是4")

            except (KeyError, TypeError) as e:
                print(f"跳过错误的元素: {e}")
            # # 转换回 BGR 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("帧已转换回 BGR 格式")
            
            # 将图像编码为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame")
            break
            
        frame = buffer.tobytes()
            #print("生成帧大小:", len(frame))

        print("发送帧...")
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()
    cv2.destroyAllWindows()

@app1.route('/video_feed')
def video_feed():
    response = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    #print("视频流已启动")
    return response



if __name__ == '__main__':
    app1.run(host='127.0.0.1', port=5001, debug=True)