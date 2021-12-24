import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from network import track_net
from focal_loss import BinaryFocalLoss
import cv2
from collections import deque
from TrackNet import ResNet_Track

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


WIDTH = 512
HEIGHT = 288

opt = keras.optimizers.Adadelta(learning_rate=1.0)

model = track_net(input_shape=(3,288,512))
#model=ResNet_Track(input_shape=(3, HEIGHT, WIDTH))

model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
load_weights = 'weights/TrackNet'
model.load_weights(load_weights)
#model.summary()

data = np.random.rand(1,3,288,512)

model.predict(data)

def main(video_path):

    
    cap = cv2.VideoCapture(video_path)

    gray_imgs = deque()

    success, image = cap.read()
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    gray_imgs.append(img)
    
    success, image = cap.read()
    #out.write(image)
    #out_2.write(image * 0)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    gray_imgs.append(img)

    cap = cv2.VideoCapture(video_path)
    y_pred = model.predict(data)

    while cap.isOpened():
        
        print('-------------------------------------')
        t0 = time.time()

        success, image = cap.read()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        gray_imgs.append(img)

        img_input = np.concatenate(gray_imgs, axis=2)
        print(len(gray_imgs))


        img_input = cv2.resize(img_input, (WIDTH, HEIGHT))
        img_input = np.moveaxis(img_input, -1, 0)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = img_input.astype('float')/255.

        y_pred = model.predict(img_input)

        y_pred = np.array(y_pred) > 0.5
        y_pred = y_pred.astype('float32')

        heapmap = np.reshape(y_pred,(HEIGHT,WIDTH,1))

        heapmap = heapmap * 255

        heapmap = cv2.resize(heapmap,dsize=(1280,720))

        heapmap = heapmap.astype('uint8')

        print(np.argmax(heapmap))

        heapmap = cv2.cvtColor(heapmap, cv2.COLOR_GRAY2BGR)

        gray_imgs.popleft()

        print(len(gray_imgs))

        cv2.imshow("heapmap",heapmap)
        cv2.imshow("image",image)

        print(1/(time.time() - t0))

        key = cv2.waitKey(1)

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':

    main('videos/test.mp4')