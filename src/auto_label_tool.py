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
import pickle

BATCH_SIZE = 1
HEIGHT=288
WIDTH=512


current = -1

parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--video_name', type=str,
                    default='tennis_FOV_2_dataset/match_2/rally_video/10.mov', help='input video name for label')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='weights/1.tar', help='input model weight for predict')
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

filename = args.video_name.split(os.sep)[-1].split('.')[0]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())

def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
            torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)

def tran_input_img(img_list):

    trans_img = []

    #for i in reversed(range(len(img_list))):
    for i in range(len(img_list)):

        img = img_list[i]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = cv2.resize(img,(WIDTH, HEIGHT))
        img = np.asarray(img).transpose(2, 0, 1) / 255.0

        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])

    trans_img = np.asarray(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])

def find_ball(pred_image, image_ori, ratio_w, ratio_h):

    if np.amax(pred_image) <= 0: #no ball
        return image_ori, 0, 0, 0, 0

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)
    # print(type(stats))

    if len(stats): 
        stats = np.delete(stats, 0, axis = 0)
        centroids = np.delete(centroids, 0, axis = 0)

    x, y , w, h, area = stats[np.argmax(stats[:,-1])]
    x_cen, y_cen = centroids[np.argmax(stats[:,-1])]

    cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
    cv2.circle(image_ori, (int(x_cen * ratio_w), int(y_cen * ratio_h)),  3, (0,0,255), -1)

    radius = int((((x + w) * ratio_w) - (x * ratio_w)) / 2)


    #for i in range(len(stats)):
    #    x, y, w, h, area = stats[i]

    return image_ori, int(x_cen * ratio_w), int(y_cen * ratio_h), radius, 1


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

input_img = []
data = dict()
racket = dict()


while cap.isOpened():


    rets = []
    images = []
    frame_times = []
    
    ret, frame = cap.read()

    current += 1

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

        h_pred = model(unit)
        torch.cuda.synchronize()

        
        h_pred = (h_pred * 255).cpu().numpy()
        
        torch.cuda.synchronize()
        h_pred = (h_pred[0]).astype('uint8')
        h_pred = np.asarray(h_pred).transpose(1, 2, 0)
        #print(h_pred.shape)

        h_pred = (50 < h_pred) * h_pred

    frame, x, y, r, visibility= find_ball(h_pred, frame, ratio_w, ratio_h)
    
    data[current] = (x, y, r)

    #h_pred = cv2.resize(h_pred, dsize=(width, height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #h_pred = cv2.resize(h_pred, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    #frame = cv2.resize(frame, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)

    #print(h_pred.shape)

    cv2.imshow("image",frame)

    #cv2.imshow("img1",input_img[0])
    #cv2.imshow("img2",input_img[1])
    #cv2.imshow("img3",input_img[2])

    #cv2.imshow("h_pred",h_pred)

    t1 = time.time()


    print("FPS : ",1/(t1 - t0))
    print((t1 - t0))

    
    if args.record:
        frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

    key = cv2.waitKey(1)

    if key == 27: break

pickle.dump([data,racket],open(filename+".pkl",'wb'))

cv2.destroyAllWindows()
#f.close()
cap.release()

if args.record:
    out.release()

