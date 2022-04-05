import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
#from utils.datasets import *
from utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(img,cfg,weights):
    
    conf_thresh = 0.4
    iou_thresh = 0.5
    augment = True
    agnostic_nms = True
    
    device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

    
    im0 = img.copy()
    img = torch.from_numpy(img) / 255.0
    
    # truncate image so that each dimension is divisible by 16
    trunc = 32
    img = img[:trunc*(img.shape[0]//trunc),:trunc*(img.shape[1]//trunc),:]
    
    img = img.permute((2,0,1)).unsqueeze(0).to(device)
    print(img.shape)

    
    # Initialize
    
    # Load model
    model = Darknet(cfg, img.shape).cuda()
    try:
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights)
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()



    # Get names and colors
    #names = load_classes()
    
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh,  agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    return pred#,names


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov4.weights', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--cfg', type=str, default='models/yolov4.cfg', help='*.cfg path')
    # parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    # opt = parser.parse_args()
    # print(opt)

    img = cv2.imread("ex_frame.png")
    cfg = 'cfg/yolov4.cfg'
    weights = 'weights/yolov4.weights'
    with torch.no_grad():
        pred = detect(img,cfg,weights)
        print(pred)
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['']:
        #         detect(img)
        #         strip_optimizer(opt.weights)
        # else:
        #     detect(img)
