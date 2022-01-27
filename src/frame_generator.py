import cv2
import csv
import os
import sys
import shutil
from glob import glob
import argparse


parser = argparse.ArgumentParser(description = 'video_trans_img')
parser.add_argument('--dataset', type = str, default='tennis_FOV_2_dataset', help = 'input your custom dataset folder path')

args = parser.parse_args()

#data_folder_list = os.listdir("./" + args.dataset)
data_folder_list = ['match_3']

print(data_folder_list)

for folder_name in data_folder_list:
	#p = os.path.join('dataset', 'tennis_FOV' ,game, '*mp4')
	p = os.path.join(args.dataset, folder_name, 'rally_video', '*mov')

	video_list = glob(p)
	print(video_list)
	#os.makedirs(folder_name + '/frame/')
	for videoName in video_list:

		path_list = videoName[:-4].split('/')
		#rallyName = videoName[len(os.path.join(folder_name, 'rally_video'))+1:-4]
		output = os.path.join('.',path_list[0] , path_list[1],'frame', path_list[3])
		
		#outputPath = os.path.join('dataset',game, 'frame', rallyName)
		#outputPath = os.path.join(args.dataset,folder_name, 'frame', output)
		
		output += '/'
		os.makedirs(output)
		cap = cv2.VideoCapture(videoName)
		success, count = True, 0
		success, image = cap.read()
		while success:
			cv2.imwrite(output + '%d.png' %(count), image)
			count += 1
			success, image = cap.read()
			print(videoName,count)
