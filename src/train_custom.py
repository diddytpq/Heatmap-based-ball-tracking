import os
import sys
import json
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from network import *
import torchvision.models as models
import dataloader_custom
import numpy as np
from dataloader_custom import TrackNetLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import math

#python train_custom.py --save_weight=custom --data_path_x=data_path_csv/FOV_2_train_list_x.csv --data_path_y=data_path_csv/FOV_2_train_list_y.csv --debug=True --epochs=10 --augmentation=True


parser = argparse.ArgumentParser(description = 'Pytorch TrackNet6')
parser.add_argument('--batchsize', type = int, default = 1, help = 'input batch size for training (defalut: 8)')
parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs to train (default: 30)')
parser.add_argument('--lr', type = float, default = 1, help = 'learning rate (default: 1)')
parser.add_argument('--tol', type = int, default = 4, help = 'tolerance values (defalut: 4)')
parser.add_argument('--optimizer', type = str, default = 'Adadelta', help = 'Adadelta or SGD (default: Adadelta)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--load_weight', type = str, default = 'weights/2.tar', help = 'the weight you want to retrain')
parser.add_argument('--save_weight', type = str, default = 'custom', help = 'the weight you want to save')
parser.add_argument('--debug', type = bool, default = False, help = 'check the predict img')
parser.add_argument('--freeze', type = bool, default = False, help = 'this option make freeze layer without last layer')
parser.add_argument('--data_path_x', type = str, default = 'data_path_csv/FOV_2_train_list_x.csv', help = 'x data path')
parser.add_argument('--data_path_y', type = str, default = 'data_path_csv/FOV_2_train_list_y.csv', help = 'y data path')
parser.add_argument('--augmentation', type = bool, default = False, help = 'this option make data augmentation')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())
train_data = TrackNetLoader(args.data_path_x, args.data_path_y, augmentation = args.augmentation)
train_loader = DataLoader(dataset = train_data, batch_size=args.batchsize, shuffle=True)

def outcome(y_pred, y_true, tol):
	n = y_pred.shape[0]
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	while i < n:
		for j in range(1):
			
			if args.debug:
				h_pred = (y_pred[i][j] * 255).cpu().numpy()
				h_true = (y_true[i][j] * 255).cpu().numpy()
				h_pred = h_pred.astype('uint8')
				h_true = h_true.astype('uint8')

				#print(h_pred.shape)
				#print(h_true.shape)


				debug_img = cv2.hconcat([h_pred, h_true])

				cv2.imshow('debug_img',debug_img)
				cv2.waitKey(1)

			if torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) == 0:
				TN += 1
			elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) == 0:
				FP2 += 1
			elif torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) > 0:
				FN += 1
			elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) > 0:

				if args.debug == False:
					h_pred = (y_pred[i][j] * 255).cpu().numpy()
					h_true = (y_true[i][j] * 255).cpu().numpy()
					h_pred = h_pred.astype('uint8')
					h_true = h_true.astype('uint8')

				#h_pred
				(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

				#h_true
				(cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
				if dist > tol:
					FP1 += 1
				else:
					TP += 1
		i += 1
	return (TP, TN, FP1, FP2, FN)

def evaluation(TP, TN, FP1, FP2, FN):
	try:
		accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		accuracy = 0
	try:
		precision = TP / (TP + FP1 + FP2)
	except:
		precision = 0
	try:
		recall = TP / (TP + FN)
	except:
		recall = 0
	return (accuracy, precision, recall)

def WBCE(y_pred, y_true):
	eps = 1e-7
	loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
	return torch.mean(loss)

def train(epoch):
	model.train()
	train_loss = 0
	TP = TN = FP1 = FP2 = FN = 0
	for batch_idx, (data, label) in enumerate(train_loader):
		t0 = time.time()

		if args.debug:
			x_0 = data[0][:3]
			x_1 = data[0][3:6]
			x_2 = data[0][6:]

			x_0 = np.asarray(x_0).transpose(1, 2, 0)
			x_1 = np.asarray(x_1).transpose(1, 2, 0)
			x_2 = np.asarray(x_2).transpose(1, 2, 0)

			input_img = cv2.hconcat([x_0, x_1, x_2])

			cv2.imshow('input_img', input_img)

		data = data.type(torch.FloatTensor).to(device)
		label = label.type(torch.FloatTensor).to(device)
		optimizer.zero_grad()
		y_pred = model(data)
		loss = WBCE(y_pred, label)
		#print('Train Epoch" {} [{}/{} ({:.0f}%)]\tLoss : {:.8f}'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader), loss.data))
		train_loss += loss.data
		loss.backward()
		optimizer.step()
		t1 = time.time()

		#print('Train Epoch" {} [{}/{} ({:.0f}%)]'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader))+'\tLoss :',format(float(loss.data.cpu().numpy()),'.1E'))
		print('Train Epoch" {} [{}/{} ({:.0f}%)]'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader))+'\tLoss :',format(float(loss.data.cpu().numpy()),'.1E'),"\t time : ",(t1 - t0),"sec")

		if(epoch % 1 == 0):
			y_pred = y_pred > 0.5
			(tp, tn, fp1, fp2, fn) = outcome(y_pred, label, args.tol)
			TP += tp
			TN += tn
			FP1 += fp1
			FP2 += fp2
			FN += fn

			#show(train_loss, epoch)
	train_loss /= len(train_loader)
	if(epoch % 1 == 0):
		display(TP, TN, FP1, FP2, FN)
		savefilename = args.save_weight + '_{}.tar'.format(epoch)
		torch.save({'epoch':epoch,'state_dict':model.state_dict(),},savefilename)
	return train_loss


def display(TP, TN, FP1, FP2, FN):
	print('======================Evaluate=======================')
	print("Number of true positive:", TP)
	print("Number of true negative:", TN)
	print("Number of false positive FP1:", FP1)
	print("Number of false positive FP2:", FP2)
	print("Number of false negative:", FN)
	(accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)
	print("Accuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print('=====================================================')

	total_accuracy_list.append(accuracy)


def show(train_loss):
	epoch_num = np.arange(1, args.epochs + 1, 1)
	plt.cla()
	plt.grid(True)
	plt.figure(figsize=(10, 8))
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title('Loss of TrackNet3')


	train_loss = train_loss

	train_loss_plt = plt.plot(epoch_num, train_loss, marker=".")
	plt.savefig('Loss_of_{}.jpg'.format(args.epochs))

	print('train loss : ' , train_loss)
	print("total_accuracy_list :",total_accuracy_list)

def dfs_freeze(model):
	for name, child in model.named_children():
		#print(name)
		if name not in ['upsample','upsample2','upsample3','upsample4','upsample5','upsample6']:
			for param in child.parameters():
				param.requires_grad = False
			dfs_freeze(child)

	return model

total_accuracy_list = []

model = efficientnet_b3()
model.to(device)
if args.optimizer == 'Adadelta':
	optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
	#optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)

if(args.load_weight):
	print('======================Retrain=======================')
	checkpoint = torch.load(args.load_weight)
	model.load_state_dict(checkpoint['state_dict'])
	epoch = checkpoint['epoch']


if args.freeze:
	model = dfs_freeze(model)


train_loss = []

for epoch in range(1, args.epochs + 1):
	loss = train(epoch)
	train_loss.append(loss.cpu().numpy())

show(train_loss)
