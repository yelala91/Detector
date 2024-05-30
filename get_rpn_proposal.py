from sys import argv
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
import mmcv
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
import numpy as np
from mmengine.registry import MODELS
import cv2

def preprocess(img, model):
    test_pipeline = get_test_pipeline_cfg(model.cfg)
    if isinstance(img, np.ndarray):
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)
        
    if isinstance(img, np.ndarray):
        data_ = dict(img=img, img_id=0)
    else:
        data_ = dict(img_path=img, img_id=0)

    data_ = test_pipeline(data_)

    data_['inputs']         = [data_['inputs']]
    data_['data_samples']   = [data_['data_samples']]
    
    data_ = model.data_preprocessor(data_)

    inputs       = data_['inputs'].cuda()
    data_samples = data_['data_samples']
    feature      = model.extract_feat(inputs)
    
    return inputs, data_samples, feature

def main():
    compare_last_result = True

    model_cfg_dict  = argv[1]
    checkpoint_path = argv[2]
    img_path        = argv[3]
    if len(argv) == 5:
        out_dir = argv[4]
    else:
        out_dir = 'out'
    num_proposal = int(argv[-1])
    
    cfg = Config.fromfile(model_cfg_dict)

    model = init_detector(cfg, checkpoint_path)
    model.test_cfg.rpn.max_per_img = num_proposal
    
    
    img = mmcv.imread(img_path)
    img_name = img_path.split('/')[-1]
    inputs, data_samples, feature = preprocess(img, model)
    # get the proposal bboxes
    rpn_results_list = model.rpn_head.predict(feature, data_samples, rescale=False)
    
    if compare_last_result:
        boxes1 = rpn_results_list[0].bboxes.detach().cpu().numpy()
        color1 = [(0, 255, 0) for i in range(boxes1.shape[0])]
        
        model.test_cfg.rpn.max_per_img = 1000
        res     = inference_detector(model, img)
        boxes2  = res.pred_instances.bboxes.detach().cpu().numpy()
        color2  = [(0, 0, 255) for i in range(boxes2.shape[0])]
        
        boxes   = [*boxes1, *boxes2]
        colors  = [*color1, *color2]
        boxes   = np.array(boxes).astype(int)
        
        for bbox, color in zip(boxes, colors):  
            x1, y1, x2, y2 = bbox  
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1) 
        
        cv2.imwrite(out_dir+'/'+img_name, img)
    else:
        boxes   = rpn_results_list[0].bboxes.detach().cpu().numpy()
        color   = 'green'
        mmcv.imshow_bboxes(img, boxes, colors=color, show=False, out_file=out_dir+'/'+img_name)
        
if __name__ == '__main__':
    main()