from mmdet.apis import init_detector, inference_detector, DetInferencer
from mmengine.config import Config
import mmcv
import sys

if __name__ == '__main__':
    cfg_path        = sys.argv[1]
    weights_path    = sys.argv[2]
    img_path        = sys.argv[3]
    if len(sys.argv) == 5:
        out_dir = sys.argv[4]
    else:
        out_dir = 'outputs/'

    inference = DetInferencer(cfg_path, weights=weights_path)

    # cfg = Config.fromfile(cfg_path)
    # model = init_detector(cfg, weights_path)
    img_name = img_path.split('/')[-1]
    
    img = mmcv.imread(img_path)
    # res = inference_detector(model, img)
    inference(img_path, out_dir=out_dir, pred_score_thr=0.6)