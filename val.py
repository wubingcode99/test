# from ultralytics import YOLO
# import torch
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
#
# model = YOLO('yolov8s-p2jct.yaml')  # 加载训练时候的模型
# # model里放训练之后的权重，       数据集放对应数据集权重
# # runs/detect/train39/weights/best.pt
# model.val(model='runs/detect/train39/weights/best.pt', data='neu.yaml', split='test')


import argparse, warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO


def transformer_opt(opt):
    opt = vars(opt)
    del opt['data']
    del opt['weight']
    return opt


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='/home/tianying/yolo/wubing/ultralytics-main/runs/detect/train27/weights/best.pt', help='training model path')
    parser.add_argument('--data', type=str, default='datasets/neu.yaml', help='data yaml path')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--batch', type=int, default=4, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='dataset split to use for validation, i.e. val, test or train')
    parser.add_argument('--project', type=str, default='runs/val', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_json', action="store_true", help='save results to JSON file')
    parser.add_argument('--save_hybrid', action="store_true",
                        help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='object confidence threshold for detection (0.001 in val)')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--half', action="store_true", help='use half precision (FP16)')
    parser.add_argument('--dnn', action="store_true", help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', action="store_true", default=True, help='ave plots during train/val')
    parser.add_argument('--rect', action="store_true", help='rectangular val')

    return parser.parse_known_args()[0]


class YOLOV8(YOLO):
    '''
    weigth:model path
    '''

    def __init__(self, weight='', task=None) -> None:
        super().__init__(weight, task)


if __name__ == '__main__':
    opt = parse_opt()

    model = YOLOV8(weight=opt.weight)
    model.val(data=opt.data, **transformer_opt(opt))
















# from ultralytics import YOLO
# # 使用测试集
# # 填写训练好的权重路径,运行前改下面的！！！！！！！！！！
# # model = YOLO('E:\\zlf\\ultralytics-main2\\runs\\detect\\train\\weights\\best.pt')
# model = YOLO('/home/tianying/yolo/GSQ/gsq/ultralytics-main/runs/detect/train39/weights/best.pt')
# # 填写数据集文件
# model.val(data='neu.yaml', batch=1, rect=False)
# # model.val(data='coco_vehicle.yaml')
