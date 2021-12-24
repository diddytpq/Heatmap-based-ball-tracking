import os
import queue
import cv2
import numpy as np
import argparse
import csv
import sys

color_list = {'red':(0,0,255), 'orange':(42,124,255), 'yellow':(1,200,255), 'blue':(255,161,0), 'pink':(186,114,228), 'green':(1,153,14)}
parser = argparse.ArgumentParser(description = 'Color, Type and Size')
parser.add_argument('--video_name', type = str, help = 'Video_name')
parser.add_argument('--csv_name', type = str, help = 'Predict csv name')
parser.add_argument('--color', type = str, default = 'red', help = 'Color (defalut: red)')
parser.add_argument('--size', type = int, default = 5, help = 'Size (default: 5)')
parser.add_argument('--type', type = int, default = -1, help = 'Type (default: -1)')
args = parser.parse_args()

input_video_path = args.video_name
input_csv_path = args.csv_name

with open(input_csv_path) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	frames = []
	x, y = [], []
	list1 = []
	for row in readCSV:
		list1.append(row)
	for i in range(1 , len(list1)):
		frames += [int(list1[i][0])]
		x += [int(float(list1[i][2]))]
		y += [int(float(list1[i][3]))]

output_video_path = input_video_path.split('.')[0] + "_trajectory.mp4"

q = queue.deque()
for i in range(0,8):
	q.appendleft(None)

#get video fps&video size
currentFrame= 0
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps*1.5, (output_width,output_height))


while(True):

	#capture frame-by-frame
	ret, img = video.read()
		#if there dont have any frame in video, break
	if not ret: 
		break
	
	if x[currentFrame] != 0 and y[currentFrame] != 0 :
		q.appendleft([x[currentFrame],y[currentFrame]])
		q.pop()
	else:
		q.appendleft(None)
		q.pop()

	for i in range(0,8):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]
			cv2.circle(img,(draw_x, draw_y), args.size, color_list[args.color], args.type)

	output_video.write(img)
	currentFrame += 1

video.release()
output_video.release()
print("finish")

