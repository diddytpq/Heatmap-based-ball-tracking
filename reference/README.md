###### tags: `CoachAI` `TrackNetV2` `GitLab`

# TrackNetV2: N-in-N-out Pytorch version (GitLab)

## :gear: 1. Install
### System Environment

- Ubuntu 18.04
- NVIDIA Gerfore GTX1080Ti
- Python 3.7.7 / git / pandas / numpy / Sklearn
- Pytorch 1.5/Opencv 4.1.0/CUDA 10.1/cudnn 7
### Package

- First, you have to install cuda, cudnn and tensorflow, tutorial:
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
- Label Tool : https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ
- Dataset link : https://drive.google.com/file/d/1JhyK49gBJUuA3RTTbTrSac5vs2kXwQdc/view?usp=sharing
- Docker file : docker pull baiduz/tracknet_pytorch:v1
        
        $ sudo apt-get install git
        $ sudo apt-get install python3-pip
        $ pip3 install pandas
        $ pip3 install opencv-python
        $ pip3 install matplotlib
        $ pip3 install -U scikit-learn
        $ pip3 install pytorch
        

<br>

## :clapper: 2. Prediction for a single video

### Generate the predicted video and the predicted labeling csv file

You can predict coordinate of shuttlecock for a single video with:

`python3 predict.py --video_name=<videoPath> --load_weight=<weightPath>`
    
Just put the video path you want to predict on option `<videoPath>` . We provide the pretrain model weights `TrackNetN_30.tar` 

#### The result should look like : 

![](https://i.imgur.com/blYMMBV.gif =700x)

### Show the predict trajectory

After `predict.py`, you will have the predicted labeling csv file. You can apply show_trajectory.py to generate ball's trajectory video for fancy purpose.

`python3 show_trajectory.py --video_name=<input_video_path> --csv_name=<input_csv_path> --color=<color> --size=<diameter of circle> --type=<thickness>`
#### For example:

![](https://i.imgur.com/GhQK07d.png)


- `<input_video_path>` = 1_01_00.mp4
- `<input_csv_path>` = 1_01_00_predict.csv
- `<color> (color of circle)` = red
- `<diameter of circle>` = 3
- `<thickness> (-1 : filled circle)` = 3
 
**After command, the result should look like :**

![](https://i.imgur.com/5xQleTI.gif)






<br>

##  :hourglass_flowing_sand: 3. Training

### Step 1 : Prepare train data
- The details of our dataset: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
- You should generate the frame data for each rally video with:

`python3 frame_generator.py`

**Your folder structure should look like :**

     profession_dataset
            ├── match1/
            │     ├── ball_trajectory/
            │     │     ├── 1_01_00_ball.csv
            │     │     ├── 1_02_00_ball.csv
            │     │     ├── …
            │     │     └── *_**_**_ball.csv
            │     ├── frame/
            │     │     ├── 1_01_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── 1_02_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── …
            │     │     └── *_**_**/
            │     │
            │     └── rally_video/
            │           ├── 1_01_00.mp4
            │           ├── 1_02_00.mp4
            │           ├── …
            │           └── *_**_**.mp4
            ├── match2/
            │ ⋮
            └── match23/
            
Apply `preprocess.py` to generate list of train data path and ground truth heatmap file.

`python3 preprocess.py`

**After command, the result should look like :**
        
    tracknet_train_list_x_N.csv
            │
            │
    tracknet_train_list_y_N.csv
            │
            │
     profession_dataset
            ├── match1/
            │     ├── ball_trajectory/
            │     │     ├── 1_01_00_ball.csv
            │     │     ├── 1_02_00_ball.csv
            │     │     ├── …
            │     │     └── *_**_**_ball.csv
            │     ├── frame/
            │     │     ├── 1_01_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── 1_02_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── …
            │     │     └── *_**_**/
            │     │
            │     ├── heatmap/
            │     │     ├── 1_01_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── 1_02_00/
            │     │     │     ├── 0.png
            │     │     │     ├── 1.png
            │     │     │     ├── …
            │     │     │     └── *.png
            │     │     ├── …
            │     │     └── *_**_**/
            │     │
            │     │
            │     └── rally_video/
            │           ├── 1_01_00.mp4
            │           ├── 1_02_00.mp4
            │           ├── …
            │           └── *_**_**.mp4
            ├── match2/
            │ ⋮
            └── match23/
            
- tracknet_train_list_x_N.csv contains list of train data image path
![](https://i.imgur.com/NyvGmxH.png)

- tracknet_train_list_y_N.csv contains list of ground truth heatmap image path
![](https://i.imgur.com/qZAGzVY.png)


- Heatmap is an ampliﬁed 2D Gaussian distribution function centered at the position of the shuttlecock center which had been labeled.
![](https://i.imgur.com/QwHpFwg.png)






### Step 2 : Start training 

`python3 train.py`

#### Flags 
- `--batchsize` : input batch size for training (defalut: 8)
- `--epochs` : number of epochs to train (default: 30)
- `--lr` : learning rate (default: 1)
- `--tol` : tolerance values of true positive (defalut: 4)
- `--optimizer` : Ada or SGD (default: Ada)
- `--momentum` : momentum fator (default: 0.9)
- `--weight_decay` : weight decay (default: 5e-4)
- `--seed` : random seed (default: 1)
- `--load_weight` : the weight you want to retrain (default: None)
- `--save_weight` : the weight name you want to save (default: TrackNet6)

**After command, the result should look like :**

![](https://i.imgur.com/x3NWcpm.png)



 
It will save intermediate weights every 3 epochs. 
:::info
If you don’t want to save intermediate weights, please comment these lines 
:::

![](https://i.imgur.com/g1iRCnl.png)



<br>

## :hammer_and_wrench: 4. How to train your own data

### Step 1 : Label ground truth data from video
We will need `_ball.csv` file generated by [our labeling tool](https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ), and put label result in ball_trajectory folder below each match folder.

     profession_dataset
            ├── match1/
            │     ├── ball_trajectory/
            │     │     ├── 1_01_00_ball.csv
            │     │     ├── 1_02_00_ball.csv
            │     │     ├── …
            │     │     └── *_**_**_ball.csv
            │     ├── frame
            │     │
            │     └── rally_video
            ├── match2/
            │ ⋮
            └── match23/


### Step 2 : Preprocess training data and start training

Just follow the step of [:hourglass_flowing_sand: 3. Training](#Step-1-:-Prepare-train-data) 
 


### Step 3 : Retrain TrackNetV2 model

If you want to retrain the model, please add load_weight argument.

`python3 train3.py --load_weight=<previousWeightPath>`

`<previousWeightPath>` is the model weights you had trained before.

Please refer to [Flags](#Flags) for other flags details.

:::info
Please cite this paper while using it : https://ieeexplore.ieee.org/abstract/document/9302757
:::

























