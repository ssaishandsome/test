from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.models.common import DetectMultiBackend

def vedio():
    device = select_device('cpu')
    model = DetectMultiBackend('F:\Yolov5-Flask-VUE-master\\back-end\weights\\best.pt', device=device, dnn=False, data=None)
    