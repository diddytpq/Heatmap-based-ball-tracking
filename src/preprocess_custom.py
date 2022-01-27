import cv2 
import csv
from glob import glob
import numpy as np
import os
import random
import pandas as pd

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

dataset = 'tennis_FOV_2_dataset'
game_list = os.listdir("./" + dataset)
#game_list = ['match_1','match_2','match_3','match_4','match_5']
game_list = ['test']
#game_list = ['match_1','match_2','match_3', 'FOV_1_match_1', 'FOV_1_match_2', 'FOV_1_match_3', 'FOV_1_match_4', 'FOV_1_match_5']

#'match1','match2','match3','match4','match5','match6','match7','match8','match9','match10']

p = os.path.join('./',dataset, game_list[0], 'frame', '1', '1.png')
print(p)
a = cv2.imread(p)
h_ratio = a.shape[0] / HEIGHT
w_ratio = a.shape[1] / WIDTH



train_x = []
train_y = []
for game in game_list:
    all_path = glob(os.path.join(dataset, game, 'frame', '*'))
    train_path = all_path[:int(len(all_path)*1)]
    for i in range(len(train_path)):
        train_path[i] = train_path[i][len(os.path.join(dataset, game, 'frame')) + 1:]
    for p in train_path:
        #p = "1"
        #print(p)
        if not os.path.exists(os.path.join(dataset, game,'heatmap',p)):
            os.makedirs(os.path.join(dataset, game,'heatmap',p))
        labelPath = os.path.join(dataset, game, 'ball_trajectory', p + '_ball.csv')
        data = pd.read_csv(labelPath)
        no = data['Frame'].values
        v = data['Visibility'].values
        x = data['X'].values
        y = data['Y'].values
        radius = data['R'].values

        num = no.shape[0]
        r = os.path.join(dataset, game, 'frame', p)
        r2 = os.path.join(dataset, game, 'heatmap', p)
        x_data_tmp = []
        y_data_tmp = []
        print(num) 
        for i in range(num-2):
            unit = []
            for j in range(3):
                target=str(no[i+j])+'.png'
                png_path = os.path.join(r, target)
                unit.append(png_path)
            print("-------------")
            print(unit)

            train_x.append(unit)
            unit = []
            
            target=str(no[i + 2])+'.png'
            heatmap_path = os.path.join(r2, target)
            if v[i + 2] == 0:
                heatmap_img = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
            else:
                round = (((radius[i + 2]) ** 2) * np.pi) / w_ratio
                heatmap_img = genHeatMap(WIDTH, HEIGHT, int(x[i+2]/w_ratio), int(y[i+2]/h_ratio), int(np.round(np.sqrt(round/np.pi))), mag)
            heatmap_img *= 255
            unit.append(heatmap_path)


            #test = heatmap_img.copy()
            #print(test.shape)
            #test = cv2.resize(test, dsize=(0, 0), fx = ratio, fy = ratio, interpolation=cv2.INTER_LINEAR)
            #cv2.imshow("test",test)
            #cv2.waitKey(0)

            cv2.imwrite(heatmap_path,heatmap_img)
            train_y.append(unit)




#input_outputfile_name = 'data_path_csv/FOV_2_train_list_x.csv'
#label_outputfile_name = 'data_path_csv/FOV_2_train_list_y.csv'

#input_outputfile_name = 'data_path_csv/gazebo_train_list_x.csv'
#label_outputfile_name = 'data_path_csv/gazebo_train_list_y.csv'

input_outputfile_name = 'data_path_csv/test_input.csv'
label_outputfile_name = 'data_path_csv/test_label.csv'

with open(input_outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s\n"%(train_x[i][0], train_x[i][1], train_x[i][2]))

with open(label_outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s\n"%(train_y[i][0]))

print('finish')

