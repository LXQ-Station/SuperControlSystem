# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import os
import time
import math
import numpy as np
import argparse
import random

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib



def eye_aspect_ratio(eye):
    # 计算眼睛的两组垂直关键点之间的欧式距离
    A = dist.euclidean(eye[1], eye[5])		# 1,5是一组垂直关键点
    B = dist.euclidean(eye[2], eye[4])		# 2,4是一组
    # 计算眼睛的一组水平关键点之间的欧式距离
    C = dist.euclidean(eye[0], eye[3])		# 0,3是一组水平关键点

    # 计算眼睛纵横比
    ear = (A + B) / (2.0 * C)
    # 返回眼睛纵横比
    return ear

def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """
    
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道

    return img_new


 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
    """
    # 判断jpg图像是否已经为4通道p
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img


