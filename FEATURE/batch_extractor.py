# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from keras.models import load_model
import h5py
import math

IMAGE_SIZE = 160 # pre-define size of image 

#====================================================================================#
# 按指定图像大小调整尺寸

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)
    
    h, w, _ = image.shape
    
    edge = max(h,w) # in order to keep the character of face, when h!=w take the long one 
    
    # paddig 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < edge:
        dh = edge - h
        top = dh // 2
        bottom = dh - top
    elif w < edge:
        dw = edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    # RGB颜色
    white = [255,255,255] # RGB 
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT is a "border type"，which means use the same color to padding
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = white)
    
    
    return cv2.resize(constant, (height, width))

#====================================================================================#

def load_dataset(data_dir):
    images = []
    labels = []
    sample_nums = [] # 用来存放不同类别的人脸数据量
    person_pics = os.listdir(data_dir) # 某一类人脸路径下的全部人脸数据文件
    print(person_pics)
    for face in person_pics:
        img = cv2.imread(os.path.join(data_dir, face)) 
        #print(img)
        
        if img is None: # 遇到部分数据有点问题，报错'NoneType' object has no attribute 'shape'，所以要判断一下图片是否为空
            pass
        else:
            img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
            
        images.append(img) # 得到某一分类下的所有图片
        labels.append(face)
    images = np.array(images)
    print(images.shape)
    labels = np.array(labels)
    return images, labels    
    
#====================================================================================#
# 进行 Embedding

def img_to_encoding(images, label, model):
    # 这里image的格式就是opencv读入后的格式, little tricks here
    '''
    cv2.imshow("image",images[4])
    cv2.waitKey(0)
    '''
    images = images[...,::-1] # Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. 这里的操作实际是对channel这一dim进行reverse，从BGR转换为RGB
    # normalization
    images = np.around(images/255.0, decimals=12) 
    #print(images)
    for n in range(len(images)):
        img=images[n]
        #print(img)
        embedding = model.predict(img[np.newaxis,:]) # predict_on_batch是对单个batch进行预测
        print(embedding.shape)
        embedding = np.squeeze(embedding)
        embedding = embedding/math.sqrt(np.dot(embedding,embedding))
        print(np.dot(embedding,embedding))
        features = np.array(embedding)
        
        print(features.shape)
        np.save("./feature_wb_{}".format(label[n]),features)
   
    return True

#====================================================================================#

if __name__ == "__main__":
    # 建立facenet模型
    facenet = load_model('./model/facenet_keras.h5') 
    #facenet.summary() #paramater of our pre-trained model
    images, labels = load_dataset('./square_WB/')
    #print(images)
    
    # create 128-Demensions features vector
    FINISH = img_to_encoding(images, labels, facenet) 

    
    
    
    
    
    

