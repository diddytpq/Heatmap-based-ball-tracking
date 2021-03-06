import os
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix()) 

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
from models.network_b0 import *

from utils import *

BATCH_SIZE = 1
HEIGHT=288
WIDTH=512

parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--video_name', type=str,
                    default='videos/test_2.mov', help='input video name for predict')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='weights/220304.tar', help='input model weight for predict')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--record', type=bool, default=False,
                    help='record option')
parser.add_argument('--tiny', type=bool,
                    default=False, help='check predict img')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())


################# video #################


cap = cv2.VideoCapture(args.video_name)
try:
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
except:
    total_frames = -1
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


ratio_h = height / HEIGHT
ratio_w = width / WIDTH
size = (width, height)

if args.record:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = args.video_name[:-4]+'_predict.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

#########################################

#f = open(args.video_name[:-4]+'_predict.csv', 'w')
#f.write('Frame,Visibility,X,Y,Time\n')

############### TrackNet ################

if args.tiny:
    model = EfficientNet_b0(1., 1.) # b3 width_coef = 1.2, depth_coef = 1.4
    checkpoint = torch.load(args.load_weight)
    model.load_state_dict(checkpoint['state_dict'])

else:
    model = EfficientNet(1.2, 1.4) # b3 width_coef = 1.2, depth_coef = 1.4
    checkpoint = torch.load(args.load_weight)
    model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()

input_img = []

start_frame = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set start frame number

while cap.isOpened():

    rets = []
    images = []
    frame_times = []
    
    ret, frame = cap.read()

    if ret == 0:
        break

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

        unit = unit / 255

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

    cv2.imshow("h_pred",h_pred)

    cv2.imshow("segment_img",segment_img)


    t1 = time.time()


    print("FPS : ",1/(t1 - t0))
    print((t1 - t0))

    
    if args.record:
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

    key = cv2.waitKey(1)

    if key == 27: break


cv2.destroyAllWindows()
#f.close()
cap.release()

if args.record:
    out.release()

