from mmdet.apis import init_detector, inference_detector, DetInferencer
from mmengine.config import Config
import mmcv

inference = DetInferencer('./model_cfgs/yolov3_cfg.py', weights='./checkpoints/yolov3_202405241958.pth')

cfg = Config.fromfile('./faster_rcnn_cfg.py')
model = init_detector(cfg, 'fr_12.pth')

img = mmcv.imread('./ren.jpg')

res = inference_detector(model, img)