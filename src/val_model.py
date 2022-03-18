import numpy as np
import torch
import os
import cv2
from models.network import *
from models.network_b0 import *

import argparse
from dataloader_custom import *

# python val_model.py --load_weight=weights/21~40/custom_11.tar

HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

TP = TN = FP1 = FP2 = FN = 0

path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='dataloader_custom')

parser.add_argument('--load_weight', type=str,
                    default='weights/220214.tar', help='input model weight for predict')

parser.add_argument('--data_path_x', type = str, 
                    default = 'data_path_csv/test_input_total.csv', help = 'x data path')
parser.add_argument('--data_path_y', type = str, 
                    default = 'data_path_csv/test_label_total.csv', help = 'y data path')

parser.add_argument('--tol', type = int, 
                    default = '8', help = 'y data path')

parser.add_argument('--debug', type=bool,
                    default=False, help='check predict img')

args = parser.parse_args()


if __name__ == '__main__' :
    print('-------------------')
    

    batchsize = 1

    test_data_path_x = args.data_path_x
    test_data_path_y = args.data_path_y

    train_data = TrackNetLoader(test_data_path_x, test_data_path_y , augmentation = False)
    train_loader = DataLoader(dataset = train_data, num_workers = 4, batch_size = batchsize, shuffle=False, persistent_workers=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())

    #model = EfficientNet(1.2, 1.4) # b3 width_coef = 1.2, depth_coef = 1.4
    model = EfficientNet_b0(1., 1.) # b3 width_coef = 1.2, depth_coef = 1.4
    
    #model = efficientnet_b3()
    model.to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    checkpoint = torch.load(args.load_weight)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    for batch_idx, (data, label) in enumerate(train_loader):

        if args.debug:
            img_0, img_1, img_2 = np.array(data[0,0:3,:,:]), np.array(data[0,3:6,:,:]), np.array(data[0,6:,:,:])

            img_0 = (img_0.transpose(1, 2, 0) * 255).astype('uint8')
            img_1 = (img_1.transpose(1, 2, 0) * 255).astype('uint8')
            img_2 = (img_2.transpose(1, 2, 0) * 255).astype('uint8')


        data = data.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            torch.cuda.synchronize()

            y_pred = model(data)

            y_true = np.array(label[0])

            y_pred = (y_pred * 255).cpu().numpy()
            y_pred = y_pred[0].astype('uint8')
            y_pred_100 = (100 < y_pred) * y_pred
            y_pred_150 = (150 < y_pred) * y_pred

                
            torch.cuda.synchronize()

            y_true = (y_true * 255).astype('uint8')


        (tp, tn, fp1, fp2, fn) = outcome(y_pred_150, y_true, args.tol)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn
        if args.debug:

            debug_img = cv2.hconcat([y_pred[0], y_true[0]])
            debug_img = cv2.hconcat([y_pred[0], y_pred_100[0], y_pred_150[0], y_true[0]])
            input_img = cv2.hconcat([img_0, img_1,img_2])

            y_jet = cv2.applyColorMap(y_pred_150[0], cv2.COLORMAP_JET)

            test = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
            test = cv2.addWeighted(test, 1, y_jet, 0.3, 0)
            test = cv2.resize(test,(1280, 720))

            cv2.imshow("input_img",input_img)
            cv2.imshow("debug_img",debug_img)
            cv2.imshow("test",test)



        display(TP, TN, FP1, FP2, FN)

        key = cv2.waitKey(1)



        if key == 27 : 
            cv2.destroyAllWindows()
            break