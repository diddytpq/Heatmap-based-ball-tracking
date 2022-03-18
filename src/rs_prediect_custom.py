import os
import sys
import json
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import numpy as np
from dataloader_custom import TrackNetLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import math
from PIL import Image
import time
from models.network import *
from utils import *

import pyrealsense2 as rs

# python predict_custom.py --load_weight=weights/21~40/custom_11.tar



BATCH_SIZE = 1
HEIGHT=288
WIDTH=512

parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--video_name', type=str,
                    default='videos/2.mov', help='input video name for predict')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='weights/220304.tar', help='input model weight for predict')
parser.add_argument('--optimizer', type=str, default='Ada',
                    help='Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--record', type=bool, default=False,
                    help='record option')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())


################# video #################

pipeline=rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
pipeline.start(config)




# cap = cv2.VideoCapture(args.video_name)
# try:
#     total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# except:
#     total_frames = -1
# fps = cap.get(cv2.CAP_PROP_FPS)
height = 360
width = 640


ratio_h = height / HEIGHT
ratio_w = width / WIDTH
size = (width, height)

# if args.record:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video_path = args.video_name[:-4]+'_predict.mp4'
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

#########################################

#f = open(args.video_name[:-4]+'_predict.csv', 'w')
#f.write('Frame,Visibility,X,Y,Time\n')

############### TrackNet ################
#model = efficientnet_b3()

model = EfficientNet(1.2, 1.4) # b3 width_coef = 1.2, depth_coef = 1.4

model.to(device)
if args.optimizer == 'Ada':
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
checkpoint = torch.load(args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()

input_img = []

start_frame = 0

#cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set start frame number

while True:

    rets = []
    images = []
    frame_times = []
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
            time.sleep(0.1)
            continue

    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    img = cv2.resize(frame,(WIDTH, HEIGHT))


    input_img.append(img)

    if len(input_img) < 3:
        continue

    if len(input_img) > 3:
        input_img = input_img[-3:]

    #unit = unit.reshape(1,9,unit.size()[-2],unit.size()[-1])
    t0 = time.time()

    unit = tran_input_img(input_img)

    unit = torch.from_numpy(unit).to(device, dtype=torch.float)
    torch.cuda.synchronize()
    
    with torch.no_grad():

        unit /= 255

        h_pred = model(unit)
        torch.cuda.synchronize()

        
        h_pred = (h_pred * 255).cpu().numpy()
        
        torch.cuda.synchronize()
        h_pred = (h_pred[0]).astype('uint8')
        h_pred = np.asarray(h_pred).transpose(1, 2, 0)
        #print(h_pred.shape)

        h_pred = (150 < h_pred) * h_pred

    segment_img = ball_segmentation(frame, h_pred, width, height)    

    frame = find_ball_v2(h_pred, frame, ratio_w, ratio_h)


    
    #h_pred = cv2.resize(h_pred, dsize=(width, height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #h_pred = cv2.resize(h_pred, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    #frame = cv2.resize(frame, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)

    #print(h_pred.shape)

    cv2.imshow("image",frame)

    #cv2.imshow("img1",input_img[0])
    #cv2.imshow("img2",input_img[1])
    #cv2.imshow("img3",input_img[2])

    #cv2.imshow("h_pred",h_pred)

    cv2.imshow("segment_img",segment_img)


    t1 = time.time()


    print("FPS : ",1/(t1 - t0))
    print((t1 - t0))

    
    if args.record:
        frame = cv2.resize(segment_img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

    key = cv2.waitKey(1)

    if key == 27: break


cv2.destroyAllWindows()
#f.close()
pipeline.stop()

if args.record:
    out.release()

