#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D
import torchfile
from tensorflow.python.layers import utils
import scipy.misc
from keras.preprocessing.image import load_img, img_to_array
import cv2 as cv


# In[2]:


def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')


# In[3]:


def encoder1_1():
  inp = Input(shape=(None, None, 3), name='vgg_input')
  t7 = torchfile.load('/wct_models/vgg_normalised_conv1_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialConvolution
  module = t7.modules[0]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #ReLU
  module = t7.modules[3]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)
  
  model = Model(inputs=inp, outputs=x)
  
  return model

def encoder2_1():
  inp = Input(shape=(None, None, 3), name='vgg_input')
  t7 = torchfile.load('/wct_models/vgg_normalised_conv2_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialConvolution
  module = t7.modules[0]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #ReLU
  module = t7.modules[3]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[7]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[10]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)   

  model = Model(inputs=inp, outputs=x)
  
  return model

def encoder3_1():
  inp = Input(shape=(None, None, 3), name='vgg_input')
  t7 = torchfile.load('/wct_models/vgg_normalised_conv3_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialConvolution
  module = t7.modules[0]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)        
  
  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)      

  #ReLU
  module = t7.modules[3]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)   
  
  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)  

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)              

  #Maxpooling
  module = t7.modules[7]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x) 

  #ReLU
  module = t7.modules[10]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x) 

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x) 

  #ReLU
  module = t7.modules[13]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x) 

  #Maxpooling
  module = t7.modules[14]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[16]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x) 

  #ReLU
  module = t7.modules[17]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x) 

  model = Model(inputs=inp, outputs=x)
  
  return model

def encoder4_1():
  inp = Input(shape=(None, None, 3), name='vgg_input')
  t7 = torchfile.load('/wct_models/vgg_normalised_conv4_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialConvolution
  module = t7.modules[0]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x) 
  
  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #ReLU
  module = t7.modules[3]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[7]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[10]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[13]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[14]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[16]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[17]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[19]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[20]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[22]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[23]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[25]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[26]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[27]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[29]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[30]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model

def encoder5_1():
  inp = Input(shape=(None, None, 3), name='vgg_input')
  t7 = torchfile.load('/wct_models/vgg_normalised_conv5_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialConvolution
  module = t7.modules[0]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #ReLU
  module = t7.modules[3]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[7]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[10]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[13]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[14]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[16]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[17]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[19]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[20]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[22]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[23]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[25]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[26]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[27]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[29]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[30]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[32]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[33]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[35]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[36]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[38]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[39]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #Maxpooling
  module = t7.modules[40]
  name = module.name.decode()[2:] if module.name is not None else None
  x = MaxPooling2D(padding='same', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x)

  #SpatialConvolution
  module = t7.modules[42]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[43]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model


# In[4]:


def decoder1_1():
  inp = Input(shape=(None, None, 64), name='vgg_input')
  t7 = torchfile.load('/wct_models/feature_invertor_conv1_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[1]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model

def decoder2_1():
  inp = Input(shape=(None, None, 128), name='vgg_input')
  t7 = torchfile.load('/wct_models/feature_invertor_conv2_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[1]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[8]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model                    

def decoder3_1():
  inp = Input(shape=(None, None, 256), name='vgg_input')
  t7 = torchfile.load('/wct_models/feature_invertor_conv3_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[1]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[8]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[13]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[15]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model   

def decoder4_1():
  inp = Input(shape=(None, None, 512), name='vgg_input')
  t7 = torchfile.load('/wct_models/feature_invertor_conv4_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[1]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[8]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[11]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[14]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[15]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[18]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[19]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[21]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[22]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[25]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[26]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[28]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)
  
  model = Model(inputs=inp, outputs=x)
  
  return model 

def decoder5_1():
  inp = Input(shape=(None, None, 512), name='vgg_input')
  t7 = torchfile.load('/wct_models/feature_invertor_conv5_1.t7',force_8bytes_long=True)
  x = inp

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[1]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[2]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[5]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[6]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[8]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[9]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[11]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[12]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[14]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[15]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[18]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[19]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[21]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[22]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[24]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[25]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[27]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[28]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[31]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[32]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[34]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[35]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #UpSampling2D
  x = UpSampling2D(size=(2, 2))(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[38]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='same', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  #ReLU
  module = t7.modules[39]
  name = module.name.decode()[2:] if module.name is not None else None
  x = Activation('relu', name=name)(x)

  #SpatialReflectionPadding
  x = Lambda(pad_reflect)(x) 

  #SpatialConvolution
  module = t7.modules[41]
  name = module.name.decode()[2:] if module.name is not None else None
  filters =module.nOutputPlane
  kernel_size = module.kH
  weight =  module.weight.transpose([2,3,1,0])
  bias = module.bias
  x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape, dtype: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape, dtype: K.constant(bias, shape=shape),
                        trainable=False)(x)

  model = Model(inputs=inp, outputs=x)
  
  return model 


# In[5]:


def wct_tf(content, style, alpha, eps=1e-8):
    '''TensorFlow version of Whiten-Color Transform
       Assume that content/style encodings have shape 1xHxWxC
       See p.4 of the Universal Style Transfer paper for corresponding equations:
       https://arxiv.org/pdf/1705.08086.pdf
    '''
    # Remove batch dim and reorder to CxHxW
    content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
    style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

    Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
    Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

    # CxHxW -> CxH*W
    content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
    style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

    # Content covariance
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

    # Style covariance
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

    # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
    with tf.device('/cpu:0'):  
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)

    ## Uncomment to perform SVD for content/style with np in one call
    ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
    # @jit
    # def np_svd(content, style):
    #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    #     Uc, Sc, _ = np.linalg.svd(content)
    #     Us, Ss, _ = np.linalg.svd(style)
    #     return Uc, Sc, Us, Ss
    # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

    # Filter small singular values
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    # Whiten content feature
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

    # Color content with style
    Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)

    # Re-center with mean of style
    fcs_hat = fcs_hat + ms

    # Blend whiten-colored feature with original content feature
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc,Hc,Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

    return blended


# In[ ]:




