import pandas as pd
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
from PIL import Image
import random
import cv2
from network import *

HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
      return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('tracknet_train_list_x_3.csv')
        label = pd.read_csv('tracknet_train_list_y_3.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('tracknet_test_list_x_3.csv')
        label = pd.read_csv('tracknet_test_list_y_3.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class TrackNetLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label_name = getData(mode)
        self.mode = mode
        img = Image.open(self.img_name[0][0]).convert('LA')
        w, h = img.size
        self.ratio = h / HEIGHT
        print("> Found %d data..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img_path = self.img_name[index]
        label_path = self.label_name[index]
        img_all = []
        label_all = []
        for i in range(3):
            x = Image.open(img_path[i]).convert('RGB')
            x = x.resize((WIDTH, HEIGHT))
            
            x = np.asarray(x).transpose(2, 0, 1) / 255.0
            #x = x.resize((WIDTH, HEIGHT, 3))

            img_all.append(x[0])
            img_all.append(x[1])
            img_all.append(x[2])
            print(x[0].shape)
        y = Image.open(label_path[-1])

        y = np.asarray(y) / 255.0
        label_all.append(y)

        img_all = np.asarray(img_all)
        label_all = np.asarray(label_all)
        '''
        if self.mode == 'train':
          if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        '''
        return img_all, label_all


if __name__ == '__main__' :
    batchsize = 1

    train_data = TrackNetLoader('' , 'train')
    train_loader = DataLoader(dataset = train_data, batch_size=batchsize, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())


    model = efficientnet_b3()
    model.to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    checkpoint = torch.load('weights/custom_5.tar')
    model.load_state_dict(checkpoint['state_dict'])


    """for batch_idx, (data, label) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor).to(device)
        y_pred = model(data)
        #print('Train Epoch" {} [{}/{} ({:.0f}%)]\tLoss : {:.8f}'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader), loss.data))
        

        print(y_pred.size())
        print(label.size())

        y_pred = (y_pred * 255).cpu().numpy()
        y_true = (label * 255).cpu().numpy()
        h_pred = y_pred.astype('uint8')
        h_true = h_true.astype('uint8')
    """

    for batch_idx, (data, label) in enumerate(train_loader):

        #print(np.array(data[0]).shape)
        #print(np.array(label[0]).shape)
        #print(type(data))

        img_0, img_1, img_2 = np.array(data[0,0:3,:,:]), np.array(data[0,3:6,:,:]), np.array(data[0,6:,:,:])

        img_0 = (img_0.transpose(1, 2, 0) * 255).astype('uint8')
        img_1 = (img_1.transpose(1, 2, 0) * 255).astype('uint8')
        img_2 = (img_2.transpose(1, 2, 0) * 255).astype('uint8')


        #img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
        #img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
        #img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)

        
        
        data = data.type(torch.FloatTensor).to(device)
        #print(type(data))
        print(data.size())
        #label = label.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            torch.cuda.synchronize()

            y_pred = model(data)

            
            #print(np.array(img_0).shape)
            #print(np.array(img_1).shape)
            #print(np.array(img_2).shape)
            y_true = np.array(label[0])
            #y_true = y_true.transpose(1, 2, 0)

            y_pred = (y_pred * 255).cpu().numpy()
            y_pred = y_pred[0].astype('uint8')
            torch.cuda.synchronize()

            y_true = (y_true * 255).astype('uint8')

            #y_pred = y_pred[0].transpose(1, 2, 0)

        #print(y_pred[0].shape)
        #print(y_true[0].shape)

        debug_img = cv2.hconcat([y_pred[0], y_true[0]])
        input_img = cv2.hconcat([img_0, img_1,img_2])


        
        cv2.imshow("input_img",input_img)
        cv2.imshow("debug_img",debug_img)



        key = cv2.waitKey(1)

        if key == 27 : 
            cv2.destroyAllWindows()
            break