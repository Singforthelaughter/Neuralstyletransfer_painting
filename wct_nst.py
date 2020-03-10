import torchfile
import tensorflow as tf
import numpy as np
import keras.backend as K
import cv2 as cv
import scipy.misc


from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D
from tensorflow.python.layers import utils
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from PIL import Image
from WCT_Function import encoder1_1, encoder2_1, encoder3_1, encoder4_1, encoder5_1, decoder1_1, decoder2_1, decoder3_1, decoder4_1, decoder5_1, wct_tf


def preprocess_image(image_path, img_height, img_width):
  img = load_img(image_path, target_size=(img_height, img_width))
  img = img_to_array(img)
  top = int(0.0014 * width)  # shape[0] = rows
  bottom = top
  left = int(0.0014 * height)  # shape[1] = cols
  right = left
  #img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, [255,255,255])
  img = np.expand_dims(img, axis=0)
  img = vgg19.preprocess_input(img)
  return img

def deprocess_image(x, hcrop, wcrop,scale_percent):
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]
  x = np.clip(x, 0, 255).astype('uint8')
  x = x[int(hcrop*x.shape[0]):int((1-hcrop)*x.shape[0]),int(wcrop*x.shape[1]):int((1-wcrop)*x.shape[1])]
  width = int(x.shape[1] * scale_percent / 100)
  height = int(x.shape[0] * scale_percent / 100)
  dim = (width, height)
  x = cv.resize(x, dim, interpolation = cv.INTER_AREA)
  #x = cv.fastNlMeansDenoisingColored(x,None,10,10,3,21)
  return x

def wct_nst(targetpath, stylepath, savepath, alpha = 0.6):
    target_image_path = targetpath
    style_reference_image_path = stylepath
    width, height = load_img(target_image_path).size

    resize = 512
    if height < width:
        ratio = height / resize
        img_width = int(width / ratio)
        img_height = resize
    else:
        ratio = width / resize
        img_height = round(height / ratio)
        img_width  = resize
    
    content_img = preprocess_image(target_image_path, img_height, img_width)
    style_img = preprocess_image(style_reference_image_path)
    a = alpha

    #Relu5_1
    encode5 = encoder5_1()
    fc = encode5.predict(content_img)
    fs = encode5.predict(style_img)
    #wct transform
    wct = wct_tf(fc,fs, alpha=a)
    #Convert tensor to array
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        wct = wct.eval()
    sess.close() 
    #Decode5_1
    decode5 = decoder5_1()
    dc5 = decode5.predict(wct)

    #Relu4_1
    encode4 = encoder4_1()
    fc = encode4.predict(dc5)
    fs = encode4.predict(style_img)
    #wct transform
    wct = wct_tf(fc,fs, alpha=a)
    #Convert tensor to array
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        wct = wct.eval()
    sess.close() 
    #Decode4_1
    decode4 = decoder4_1()
    dc4 = decode4.predict(wct)

    #Relu3_1
    encode3 = encoder3_1()
    fc = encode3.predict(dc4)
    fs = encode3.predict(style_img)
    #wct transfrom
    wct = wct_tf(fc,fs, alpha=a)
    #Convert tensor to array
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        wct = wct.eval()
    sess.close() 
    #Decode3_1
    decode3 = decoder3_1()
    dc3 = decode3.predict(wct)

    #Relu2_1
    encode2 = encoder2_1()
    fc = encode2.predict(dc3)
    fs = encode2.predict(style_img)
    #wct transfrom
    wct = wct_tf(fc,fs, alpha=a)
    #Convert tensor to array
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        wct = wct.eval()
    sess.close() 
    #Decode2_1
    decode2 = decoder2_1()
    dc2 = decode2.predict(wct)

    #Relu1_1
    encode1 = encoder1_1()
    fc = encode1.predict(dc2)
    fs = encode1.predict(style_img)
    #wct transfrom
    wct = wct_tf(fc,fs, alpha=a)
    #Convert tensor to array
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        wct = wct.eval()
    sess.close() 
    #Decode1_1
    decode1 = decoder1_1()
    dc1 = decode1.predict(wct)

    final = tf.squeeze(dc1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        final = final.eval()
    sess.close() 

    img = deprocess_image(final,0.17,0.12,100)

    plt.imsave(savepath, img)