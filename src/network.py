import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def res_block_down_sample(filters, strides, inputs, **conv_kwargs):
    
    if strides == 2:
        
        short_cut = keras.layers.AveragePooling2D((2, 2), strides=strides, padding='same', data_format='channels_first')(inputs)
        short_cut = keras.layers.Conv2D(filters * 2, (1,1), strides=1, padding='same', data_format='channels_first')(short_cut)
        short_cut = keras.layers.BatchNormalization()(short_cut)
    
    else:
        
        short_cut = inputs
        
    
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, (1,1), strides = 1, padding = 'same',data_format = 'channels_first', **conv_kwargs)(x)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, (3,3), strides = strides, padding = 'same',data_format = 'channels_first', **conv_kwargs)(x)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters * 2, (1,1), strides = 1, padding = 'same',data_format = 'channels_first', **conv_kwargs)(x)

        
    x = keras.layers.add([x, short_cut])
    
    return x


def res_block_up_sample(filters, strides, inputs, transpose = False, **conv_kwargs):
    
    if transpose:
        
        short_cut = keras.layers.UpSampling2D((2,2), interpolation = 'bilinear', data_format = 'channels_first')(inputs)
        short_cut = keras.layers.Conv2D(filters, (1,1), strides = 1, padding = 'same',data_format = 'channels_first')(short_cut)
        short_cut = keras.layers.BatchNormalization()(short_cut)
        
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters, (1,1), strides = 1, padding = 'same',data_format = 'channels_first',  **conv_kwargs)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, (3,3), strides = strides, padding = 'same',data_format = 'channels_first', output_padding = 1,  **conv_kwargs)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters, (1,1), strides = 1, padding = 'same',data_format = 'channels_first',  **conv_kwargs)(x)
    
    
    else:
        
        short_cut = keras.layers.Conv2D(filters, (1,1), strides=1, padding='same', data_format='channels_first')(inputs)
               
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters, (1,1), strides = 1, padding = 'same',data_format = 'channels_first', **conv_kwargs)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters, (3,3), strides = strides, padding = 'same',data_format = 'channels_first', **conv_kwargs)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters, (1, 1), strides=1, padding="same", data_format='channels_first', **conv_kwargs)(x)


    
    x = keras.layers.add([x, short_cut])
    
    return x
    

def track_net(input_shape):
    #start_layers
    inputs = keras.Input(shape = input_shape, name = "input")
    x = keras.layers.Conv2D(64, (3,3), padding = 'same', data_format = 'channels_first')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    x = keras.layers.Conv2D(64, (3,3), padding = 'same', data_format = 'channels_first')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    #res_block_down_sample_1
    x = res_block_down_sample(filters = 16, strides = 2, inputs = x)
    x = res_block_down_sample(filters = 16, strides = 1, inputs = x)
    x_1 = res_block_down_sample(filters = 16, strides = 1, inputs = x)
    
    #res_block_down_sample_2
    x = res_block_down_sample(filters = 32, strides = 2, inputs = x_1)
    x = res_block_down_sample(filters = 32, strides = 1, inputs = x)
    x_2 = res_block_down_sample(filters = 32, strides = 1, inputs = x)
    
    #res_block_down_sample_3
    x = res_block_down_sample(filters = 64, strides = 2, inputs = x_2)
    x = res_block_down_sample(filters = 64, strides = 1, inputs = x)
    x = res_block_down_sample(filters = 64, strides = 1, inputs = x)
    x_3 = res_block_down_sample(filters = 64, strides = 1, inputs = x)
    
    #res_block_down_sample_4
    x = res_block_down_sample(filters = 128, strides = 2, inputs = x_3)
    x = res_block_down_sample(filters = 128, strides = 1, inputs = x)
    x = res_block_down_sample(filters = 128, strides = 1, inputs = x)   
       
    #res_block_up_sample_1
    x = res_block_up_sample(filters = 128, strides = 2, inputs = x, transpose = True)
    x = tf.concat([x,x_3],axis=1)
    
    x = res_block_up_sample(filters = 128, strides = 1, inputs = x, transpose = False)
    x = res_block_up_sample(filters = 128, strides = 1, inputs = x, transpose = False)
    x = res_block_up_sample(filters = 128, strides = 1, inputs = x, transpose = False)
    
    #res_block_up_sample_2
    x = res_block_up_sample(filters = 64, strides = 2, inputs = x, transpose = True)
    x = tf.concat([x,x_2],axis=1)
    
    x = res_block_up_sample(filters = 64, strides = 1, inputs = x, transpose = False)
    x = res_block_up_sample(filters = 64, strides = 1, inputs = x, transpose = False)
    
    #res_block_up_sample_3
    x = res_block_up_sample(filters = 32, strides = 2, inputs = x, transpose = True)
    x = tf.concat([x,x_1],axis=1)
    
    x = res_block_up_sample(filters = 32, strides = 1, inputs = x, transpose = False)
    x = res_block_up_sample(filters = 32, strides = 1, inputs = x, transpose = False)
    
    #res_block_up_sample_4
    x = res_block_up_sample(filters = 16, strides = 2, inputs = x, transpose = True)
    
    #end_layers
    x = keras.layers.Conv2D(64, (3,3), padding = 'same', data_format = 'channels_first')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(64, (3,3), padding = 'same', data_format = 'channels_first')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(256, (3,3), padding = 'same', data_format = 'channels_first', bias_initializer = keras.initializers.constant(-3.2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    outputs = keras.layers.Activation("softmax")(x)

    #outputs = x
    
    outputs = tf.reduce_max(outputs, axis=1)
    outputs = tf.expand_dims(outputs, axis=1)

    return keras.Model(inputs, outputs, name="track_net")
        
    

if __name__ == "__main__":

    model = track_net(input_shape=(3,288,512))
    model.summary()

    data = np.random.rand(1,3,288,512)

    time_list = list()

    for i in range(40):
        t0 = time.time()
        a = model.predict(data)

        time_list.append(1/(time.time() - t0))
        
    print("avg time : ",sum(time_list)/len(time_list))