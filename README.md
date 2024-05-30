# 基于 MMDetection 框架训练目标检测模型 Faster R-CNN和 YOLO v3

## 所需依赖
```
完整的 MMDetection>=3.0 框架
```

## 训练

### 数据集准备

在项目根目录新建文件夹 ```data/coco/voc0712/annotations``` 和 ```data/coco/voc0712/VOCdevkit``` 下载 PASCAL2007 和 PASCAL2012, 解压后将 ```VOC2007``` 和 ```VOC2012``` 两个文件夹复制到目录 ```data/coco/voc0712/VOCdevkit``` 下, 然后终端运行如下命令将数据转换成coco格式
```
python tools/dataset_converters/pascal_voc.py \
                  data/coco/voc0712/VOCdevkit \
             -o data/coco/voc0712/annotations \
                            --out-format coco
```

### 开始训练

在终端运行

```
python tools/train.py model_cfgs/faster_rcnn_cfg.py
```

如果要训练 YOLO v3 将上述命令中的 ```faster_rcnn_cfg.py``` 替换成 ```yolov3_cfg.py``` 即可.

## 测试

如果要用训练好的权重加载到模型中去推理一张图片, 执行如下命令
```
python infer.py ${模型的配置文件地址}$ ${权重地址}$ ${图片地址}$ ${输出结果的地址}$
```
其中 模型的配置文件地址 在这里分别是 ```model_cfgs/faster_rcnc_cfg.py``` 或 ```model_cfgs/yolov3_cfg.py```.

## 获取 RPN 生成的 proposal box

如果要获取 Faster R-CNN 中 RPN 网络生成的 proposal box 可以执行如下命令
```
python get_rpn_proposal.py model_cfgs/faster_rcnn_cfg.py \
                                             ${权重地址}$ \
                                             ${图片地址}$ \
                                        ${输出结果的地址}$ \
                                                       20 
```

其中最后一个参数是最终绘制出来的 proposal box 的个数

