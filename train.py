from ultralytics import YOLO
import torch

def main():
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # EfficientNetV2, MobileNetV3, PP_LCNet, Fasternet, EfficientVit, VanillaNet
    # can run: VanillaNet,EfficientNetV2,FasterNet,MobileNetV3,PP_LCNet,MobileVitV3_XS
    print('加载模型')
    model = YOLO('yolov8_att.yaml')  # build from YAML and transfer weights

    print('开始训练模型')
    #
    # Train the model 训练   coco128
    model.train(data='VisDrone2019.yaml', epochs=2, imgsz=640, batch=16, rect=False)
    # 测试集
    # model.val(model='yolov8s.pt', data='voc_vehicle.yaml', split='test')
    # #yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640


if __name__ == '__main__':
    main()
    
