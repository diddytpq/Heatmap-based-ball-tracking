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
from network import *


BATCH_SIZE = 1
HEIGHT=288
WIDTH=512


parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--video_name', type=str,
                    default='videos/test.mp4', help='input video name for predict')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='weights/custom_5.tar', help='input model weight for predict')
parser.add_argument('--optimizer', type=str, default='Ada',
                    help='Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())


def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
            torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)


def custom_time(time):
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    s = remain % 60
    s += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    #Generate custom time string
    cts = ''
    if len(str(h)) >= 2:
        cts += str(h)
    else:
        for i in range(2 - len(str(h))):
            cts += '0'
        cts += str(h)

    cts += ':'

    if len(str(m)) >= 2:
        cts += str(m)
    else:
        for i in range(2 - len(str(m))):
            cts += '0'
        cts += str(m)

    cts += ':'

    if len(str(int(s))) == 1:
        cts += '0'
    cts += str(s)

    return cts

def tran_input_img(img_list):

    trans_img = []

    for i in range(len(img_list)):

        img = img_list[i]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(WIDTH, HEIGHT))
        img = np.asarray(img).transpose(2, 0, 1) / 255.0


        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])

    trans_img = np.asarray(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])



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

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = args.video_name[:-4]+'_predict.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

#########################################

#f = open(args.video_name[:-4]+'_predict.csv', 'w')
#f.write('Frame,Visibility,X,Y,Time\n')

############### TrackNet ################
model = efficientnet_b3()
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
#model.eval()
count = 0
count2 = -3
time_list = []
start1 = time.time()
input_img = []
while cap.isOpened():
    rets = []
    images = []
    frame_times = []
    
    t0 = time.time()
    ret, frame = cap.read()


    input_img.append(frame)

    if len(input_img) < 3:
        continue

    if len(input_img) > 3:
        input_img = input_img[-3:]

    #unit = unit.reshape(1,9,unit.size()[-2],unit.size()[-1])

    unit = tran_input_img(input_img)
    unit = torch.from_numpy(unit).to(device, dtype=torch.float)
    
    with torch.no_grad():
        t0 = time.time()

        torch.cuda.synchronize()
        start = time.time()
        h_pred = model(unit)
        torch.cuda.synchronize()

        t1 = time.time()

        end = time.time()
        time_list.append(end - start)
        
        h_pred = (h_pred * 255).cpu().numpy()
        
        torch.cuda.synchronize()
        h_pred = (h_pred[0]).astype('uint8')
        #h_pred = (200 < h_pred) * h_pred


    """for idx_f, (image, frame_time) in enumerate(zip(images, frame_times)):
        #show = np.copy(image)
        show = cv2.resize(image, (frame.shape[1], frame.shape[0]))
        # Ball tracking
        
        if np.amax(h_pred[idx_f]) <= 0:  # no ball
        #if torch.amax(h_pred[idx_f]) <= 0:  # no ball
            #f.write(str(count2 + (idx_f))+',0,0,0,'+frame_time+'\n')
            #out.write(image)
            pass
        else:
            (cnts, _) = cv2.findContours(h_pred[idx_f], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for i in range(len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
            (cx_pred, cy_pred) = (int(ratio_w*(target[0] + target[2] / 2)), int(ratio_h*(target[1] + target[3] / 2)))
            
            cv2.circle(frame, (cx_pred, cy_pred), 5, (0,0,255), -1)
            print((cx_pred, cy_pred))
            
            #f.write(str(count2 + (idx_f))+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
            
            #out.write(image)"""


    #heapmap = np.reshape(h_pred,(HEIGHT,WIDTH,1))

    #print(heapmap.shape)

    #heapmap = cv2.cvtColor(heapmap, cv2.COLOR_GRAY2BGR)
    #heapmap = cv2.resize(heapmap, dsize=(1280, 720), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #print(h_pred.shape)
    cv2.imshow("image",frame)
    cv2.imshow("h_pred",h_pred[0])


    print("FPS : ",1/(t1 - t0))
    print((t1 - t0))

    #print(1/(end - start))


    key = cv2.waitKey(1)

    if key == 27: break


cv2.destroyAllWindows()
#f.close()
cap.release()
out.release()
end1 = time.time()
print('Prediction time:', (end1-start1), 'secs')
print('FPS', total_frames / (end1-start1) )
print('Done......')
