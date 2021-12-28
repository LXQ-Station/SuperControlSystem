# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
#from tensorflow import keras
from keras.models import load_model
import h5py

import cv2
import mediapipe as mp
import os
import time
import math
import numpy as np
import argparse
import random
from FaceFHE import *
#from feature_extract import * 



parser = argparse.ArgumentParser()
parser.add_argument('--user', default='visiter', help='who are you')
opt = parser.parse_args()
print(opt)


if __name__ == "__main__":

    control = HandControlVolume()
    control.recognize()
    
    '''
    images, labels = load_dataset('./dataset_image/'+opt.user)
    
    # create 128-Demensions features vector
    X_embeddings = img_to_encoding(images, facenet) 
    #print(X_embeddings.shape)
    
    features = np.array(X_embeddings)
  
    print(features.shape)
    np.save("./dataset_image/{}_feature".format(opt.user), features)    
    '''
 
