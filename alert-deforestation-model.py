#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
import tensorflow.keras.backend as K
from osgeo import gdal

from matplotlib import pyplot as plt
import random
from skimage.io import imshow
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

seed = 101

IMAGE_HEIGHT = IMAGE_WIDTH = 256
NUM_BANDS = 10
images_path = 'Data/BuildingsDataSet/Images/'
masks_path = 'Data/BuildingsDataSet/Masks/'

def load_image(image):
    return gdal.Open(image, gdal.GA_ReadOnly)
    
def convert_to_array(dataset):
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(bands, 2)

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

files = next(os.walk(images_path))[2]

all_images = []
all_masks = []
num_skipped = 0
for image_file in files:

    image_path = images_path + image_file
    image = load_image(image_path)
    image_data = convert_to_array(image)
    image_data[np.isnan(image_data)] = 0 # adiciona 0 onde é NaN
    image_data = normalize(image_data)

    mask_path = masks_path + image_file
    mask = load_image(mask_path)
    mask_data = convert_to_array(mask)
    mask_data[np.isnan(mask_data)] = 0 # adiciona 0 onde é NaN
    mask_data[mask_data>0] = 1
    mask_data[mask_data<=0] = 0
    
    # Pule qualquer imagem que esteja mais de 99% vazia.
    if np.any(mask_data):
        num_white_pixels = len(mask_data[mask_data==1])
        num_black_pixels = len(mask_data[mask_data==0])
        if num_black_pixels == 0: num_black_pixels = 1 # para evitar erro de dividir por 0

        if num_white_pixels/num_black_pixels < 0.01:
            num_skipped+=1
            continue
                   
        all_images.append(image_data)
        all_masks.append(mask_data)
    else: 
        num_skipped+=1
    
    if len(all_images) >= 20:
        break

images = np.array(all_images)
masks = np.array(all_masks, dtype=int)
print('Total imagens: \n', len(all_images))
print('Images: \n', images.shape)
print('Masks: \n', masks.shape)
print("\n{} Images were skipped.".format(num_skipped))
print("\nUnique elements in the train mask:", np.unique(masks))

# ### Defining Custom Loss functions.

#Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# ### Splitting data

train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=seed)
del images, masks
print("TRAIN SET")
print(train_images.shape)
print(train_masks.shape)
print("TEST SET")
print(test_images.shape)
print(test_masks.shape)


# ### Display exemple image

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


i = 1
sample_image, sample_mask = train_images[i], train_masks[i]
display([sample_image[:,:,[2,1,0]], sample_mask])


# ### Importing Our Model

import importlib
import import_ipynb

# para recarregar do modelo quando tiver alguma alteração
#importlib.reload(M)

import unet_lstm_model as M

model = M.unet_filters_32x256(input_size = (IMAGE_WIDTH, IMAGE_HEIGHT, NUM_BANDS))
model.summary()


# ### Show prediction image exemple

def show_predictions(dataset=None, num=1):
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    display([sample_image[:,:,[2,1,0]], sample_mask, pred_mask[0]])
    
show_predictions()


# ### Hyper parameters

EPOCHS = 5
LEARNING_RATE = 0.0001
BATCH_SIZE = 16


# ### Callbacks

model_path = "Models/checkpoint_02.h5"
checkpointer = ModelCheckpoint(model_path,
                               monitor="val_loss",
                               mode="min",
                               save_best_only = True,
                               #save_weights_only=True,
                               verbose=1
                              )
"""
earlystopper = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)
"""

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               mode="auto"
                              )

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


# ### Compiling the model

opt = optimizers.Adam(LEARNING_RATE)
model.compile(
      optimizer=opt,
      loss=soft_dice_loss,
      metrics=[iou_coef, 'acc'])

history = model.fit(train_images,
                    train_masks/1,
                    validation_split = 0.1,
                    epochs=EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks=[checkpointer, lr_reducer, DisplayCallback()]
                  )

"""model.save("./Models/deforestation_trained_model_02.h5")


# ## Testing our Model

model = load_model("./Models/deforestation_trained_model_02.h5", custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})


model.evaluate(test_images, test_masks)

predictions = model.predict(test_images, verbose=1)

i = 0
p = predictions[i]
p = np.where(p > 0.5, 1, 0)

f = plt.figure(figsize=(20,20))

f.add_subplot(3, 3, 1)
# 2 = B4, 1 = B3, 0 = B2
plt.imshow(tf.keras.preprocessing.image.array_to_img(test_images[i][:,:,[2,1,0]]))
plt.title('Image')

f.add_subplot(3, 3, 2)
plt.imshow(test_masks[i])
plt.title('Mask')

f.add_subplot(3, 3, 3)
plt.imshow(p)
plt.title('Prediction')
       
plt.show()
"""