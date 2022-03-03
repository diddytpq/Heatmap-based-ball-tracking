import numpy as np
import os
from PIL import Image
import random
import cv2
from network import *
import math
import albumentations as A
import argparse

from dataloader_custom import *

HEIGHT=288
WIDTH=512


parser = argparse.ArgumentParser(description = 'video_trans_img')
parser.add_argument('--dataset', type = str, default='tennis_FOV_dataset', help = 'input your custom dataset folder path')
parser.add_argument('--data_path_x', type = str, default = 'data_path_csv/tracknet_train_list_x.csv', help = 'this option make freeze layer without last layer')
parser.add_argument('--data_path_y', type = str, default = 'data_path_csv/tracknet_train_list_y.csv', help = 'this option make freeze layer without last layer')

args = parser.parse_args()

def main():
    count = 2

    train_x = []
    train_y = []

    data_folder_list = os.path.join("./" , args.dataset)

    input_data = data_folder_list + '/augment_data' + '/frame/1'
    label_data = data_folder_list + '/augment_data' + '/heatmap/1'

    os.makedirs(input_data ,exist_ok=True)
    os.makedirs(label_data ,exist_ok=True)

    x_data_name_list, y_data_name_list = getData(data_path_x = args.data_path_x, data_path_y = args.data_path_y)

    #for i in range(len(x_data_name_list)):
    for i in range(100):

        """transform = A.Compose([
            #A.RandomCrop(width=256, height=256),
            #A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.9),
            ])"""
        transform = A.Compose([
                    A.HorizontalFlip(p=0.3),
                    A.Cutout(num_holes=30, max_h_size = 16, max_w_size = 16, fill_value=0, p = 0.3),
                    #A.RandomContrast(limit=0.8, p=0.5),
                    #A.ShiftScaleRotate(scale_limit=0.50, rotate_limit=45, p=.5),
                    A.Rotate(limit = (-45,45), p = 0.4),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit = 0.2, p=0.3),
                    A.Blur(blur_limit=(3, 3), p =0.2),
                    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.2),
                    #A.ChannelShuffle(p=0.3)
                ],
                additional_targets = {
                    'image1' :'image',
                    'image2' :'image'
                })

        img_0 = cv2.imread(x_data_name_list[i][0])
        img_1 = cv2.imread(x_data_name_list[i][1])
        img_2 = cv2.imread(x_data_name_list[i][2])

        img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)


        heatmap_img = cv2.imread(y_data_name_list[i])
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

        img = [img_0, img_1, img_2]

        transformed = transform(image = img_0, image1= img_1, image2 = img_2, mask = heatmap_img)
        #transformed = transform(image = img_0, mask = heatmap_img)
        transformed_image_0 = transformed['image']
        transformed_image_1 = transformed['image1']
        transformed_image_2 = transformed['image2']

        transformed_masks = transformed['mask']

        transformed_image_0 = cv2.cvtColor(transformed_image_0, cv2.COLOR_RGB2BGR)
        transformed_image_1 = cv2.cvtColor(transformed_image_1, cv2.COLOR_RGB2BGR)
        transformed_image_2 = cv2.cvtColor(transformed_image_2, cv2.COLOR_RGB2BGR)
        transformed_masks = cv2.cvtColor(transformed_masks, cv2.COLOR_RGB2GRAY)


        input_img = cv2.hconcat([img_0, img_1, img_2])
        trans_img = cv2.hconcat([transformed_image_0, transformed_image_1, transformed_image_2])

        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2GRAY)

        debug_img = cv2.hconcat([heatmap_img, transformed_masks])

        cv2.imshow("input_img", input_img)
        cv2.imshow("trans_img", trans_img)
        cv2.imshow("debug_img", debug_img)

        cv2.imwrite(input_data + '/%d.png' %(count-2), transformed_image_0)
        cv2.imwrite(input_data + '/%d.png' %(count-1), transformed_image_1)
        cv2.imwrite(input_data + '/%d.png' %(count), transformed_image_2)
        cv2.imwrite(label_data + '/%d.png' %(count), transformed_masks)

        png_path_0 = input_data[2:] + '/' + str(count-2) + '.png'
        png_path_1 = input_data[2:] + '/' + str(count-1) + '.png'
        png_path_2 = input_data[2:] + '/' + str(count) + '.png'

        label_path = label_data[2:] + '/' + str(count) + '.png'

        train_x.append([png_path_0, png_path_1, png_path_2])
        train_y.append([label_path])




        count += 3
        key = cv2.waitKey(1)



        if key == 27:
            cv2.destroyAllWindows()
            break

    return train_x, train_y
if __name__ == '__main__':

    train_x, train_y = main()

    input_outputfile_name = 'data_path_csv/augment_input.csv'
    label_outputfile_name = 'data_path_csv/augment_label.csv'

    with open(input_outputfile_name,'w') as outputfile:
        for i in range(len(train_x)):
            outputfile.write("%s,%s,%s\n"%(train_x[i][0], train_x[i][1], train_x[i][2]))

    with open(label_outputfile_name,'w') as outputfile:
        for i in range(len(train_x)):
            outputfile.write("%s\n"%(train_y[i][0]))