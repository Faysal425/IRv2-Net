from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class IRv2_Net:
 
    def __init__(self, input_size=256):
        self.input_size = input_size
    
    def build_model(self):
        def conv_block(input, num_filters):
            x = Conv2D(num_filters, 3, padding="same")(input)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            return x
        
        def decoder_block(input, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
            x = Concatenate()([x, skip_features])
            x = conv_block(x, num_filters)
            return x
        
        """ Input """
        inputs = Input((self.input_size, self.input_size, 3))
        
        """ Pre-trained InceptionResNetV2 Model """
        encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
        
        """ Encoder """
        s1 = encoder.layers[3].output  # Changed from [0] to [3]
        s1 = ZeroPadding2D(((0, 1), (0, 1)))(s1)  # Adjusted padding
        
        s2 = encoder.layers[10].output  # Changed from [3] to [10]
        s2 = ZeroPadding2D(((1, 1), (1, 1)))(s2)  # Adjusted padding
        
        s3 = encoder.layers[280].output  # Changed from [13] to [280]
        s3 = ZeroPadding2D(((1, 2), (1, 2)))(s3)  # Adjusted padding
        s3 = Conv2D(256, 1, padding='same')(s3)  # Added filter expansion
        
        s4 = encoder.layers[594].output  # Changed from [265] to [594]
        s4 = ZeroPadding2D(((1, 1), (1, 1)))(s4)  # Adjusted padding
        
        """ Bridge """
        b1 = encoder.layers[780].output  # Changed from [501] to [780]
        b1 = ZeroPadding2D(((1, 1), (1, 1)))(b1)  
        
        """ Decoder """
        d1 = decoder_block(b1, s4, 512)  
        d2 = decoder_block(d1, s3, 256)  
        d3 = decoder_block(d2, s2, 128)  
        d4 = decoder_block(d3, s1, 64)  
        
        """ Output """
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        
        model = Model(inputs, outputs, name="IRv2-Net")
        return model