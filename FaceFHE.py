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

from feature_extract import *
from utils import *


# 方块管理类

class SquareManager:
    def __init__(self,rect_width):
        # 方框长度
        self.rect_width = rect_width
        
        # 方块list
        self.square_count = 0
        self.rect_left_x_list = []
        self.rect_left_y_list = []
        self.alpha_list = []

        # 中指与矩形左上角点的距离
        self.L1 = 30
        self.L2 = 30

        # 激活移动模式
        self.drag_active_move = False
        self.drag_active_clic = False
        # 激活的方块ID
        self.active_index = -1
        
        
    # 创建一个方块，但是没有显示
    def create(self,rect_left_x,rect_left_y,alpha=0.4):
        self.rect_left_x_list.append(rect_left_x)
        self.rect_left_y_list.append(rect_left_y)
        self.alpha_list.append(alpha)
        self.square_count+=1
        

    # 更新位置
    def display(self,class_obj,radius):

       icon_collect = []
       icon1 = cv2.imread("icon/face.png", cv2.IMREAD_UNCHANGED)
       icon2 = cv2.imread("icon/capture.png", cv2.IMREAD_UNCHANGED)
       icon3 = cv2.imread("icon/start.PNG", cv2.IMREAD_UNCHANGED)
       icon4 = cv2.imread("icon/stop.PNG", cv2.IMREAD_UNCHANGED)

       icon_collect.append(icon1)  
       icon_collect.append(icon2) 
       icon_collect.append(icon3)
       icon_collect.append(icon4)

       overlay = class_obj.image.copy() # copy 一个

       for i in range(0,self.square_count):
            x= self.rect_left_x_list[i]
            y= self.rect_left_y_list[i]
            alpha  = self.alpha_list[i]
  
            icon1 = cv2.resize(icon_collect[i], (100,100), interpolation=cv2.INTER_AREA)
            icon2 = cv2.resize(icon_collect[i], (radius-20,radius-20), interpolation=cv2.INTER_AREA)
            if(i == self.active_index):
                overlay = merge_img(overlay, icon2, y, y+radius-20, x, x+radius-20)             
            else:
                overlay = merge_img(overlay, icon1, y, y+100, x, x+100)
                              
            class_obj.image = overlay
            

    # 判断落在哪个方块上，返回方块的ID
    def checkOverlay(self,check_x,check_y):
        for i in range(0,self.square_count):
            x= self.rect_left_x_list[i]
            y= self.rect_left_y_list[i]

            if (x <  check_x < (x+self.rect_width) ) and ( y < check_y < (y+self.rect_width)):
                
                # 保存被激活的方块ID
                self.active_index = i
                return i
        return -1


    # 计算与指尖的距离
    def setLen(self,check_x,check_y):
        # 计算距离
        self.L1 = check_x - self.rect_left_x_list[self.active_index] 
        self.L2 = check_y - self.rect_left_y_list[self.active_index]


    # 更新方块    
    def updateSquare(self,new_x,new_y):
        # print(self.rect_left_x_list[self.active_index])
        self.rect_left_x_list[self.active_index] = new_x - self.L1 
        self.rect_left_y_list[self.active_index] = new_y - self.L2

#==============================================================================================================================#

# 识别控制类型
class HandControlVolume:
    def __init__(self):
        # initialisation of medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # 中指与矩形左上角点的距离
        self.L1 = 0
        self.L2 = 0

        # image实例，以便另一个类调用
        self.image=None
                

    # 主函数
    def recognize(self):
        # 计算刷新率
        fpsTime = time.time()
        # OpenCV读取视频流
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 1280
        resize_h = 720
        # 画面显示初始化参数
        rect_percent_text = 0
        radius = 0
        # 初始化方块管理器
        squareManager = SquareManager(80)
        # 创建多个方块
        for i in range(0,2):
            #squareManager.create(200*i+150,550,0.8)
            squareManager.create(1000,170*i+190,0.8)
        for i in range(0,2):
            #squareManager.create(200*i+150,550,0.8)
            squareManager.create(30,170*i+300,0.8)


        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=1) as hands:
                                 
            # ======================== [initialisation of parameter] ======================== #
            
            pause=-5
            face=0
            photo=0
            take = False
            eye = 1
            eye_margin = 0.2 #0.27
            eye_count = 0
            mouth = 0
            mouth_margin = 0.8
            
            # ======================== [initialisation of parameter] ======================== #
            
            '''
            # 表示脸部位置检测器
            detector = dlib.get_frontal_face_detector()
            # 表示脸部特征位置检测器
            predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
            # 左右眼的索引
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            # 嘴唇的索引
            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            '''
            
            while cap.isOpened():
            
                # 初始化矩形
                success, self.image = cap.read()
                self.image = cv2.resize(self.image, (resize_w, resize_h))
                square = self.image[210:210+400,450:450+380] # 对图片进行剪切
                gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                if photo!=0 and take:
                    folder = os.path.exists("./dataset_image/consensus/" + opt.user)
                    if not folder:                   #判断是否存在文件夹如果不存在则创建
                        os.makedirs("./dataset_image/consensus/" + opt.user)
                    cv2.imwrite("./dataset_image/consensus/{}/{}.jpg".format(opt.user, random.randint(11111,99999)), gray)
                if photo!=0 and take:
                    folder = os.path.exists("./dataset_image/color/" + opt.user)
                    if not folder:                   #判断是否存在文件夹如果不存在则创建
                        os.makedirs("./dataset_image/color/" + opt.user)
                    cv2.imwrite("./dataset_image/color/{}/{}.jpg".format(opt.user, random.randint(11111,99999)), square)
                    
                if face==1:
                    mp_face_detection = mp.solutions.face_detection
                    mp_drawing = mp.solutions.drawing_utils
                    results = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7).process(self.image)
                    take = False
                    if results.detections:
                        for detection in results.detections:
                            X=detection.location_data.relative_keypoints[2].x
                            print("x = ",X)
                            Y=detection.location_data.relative_keypoints[2].y
                            print("Y = ",Y)
                            #if X>600 and X<680 and Y>300 and Y<420:
                            if X>0.45 and X<0.55 and Y>0.55 and Y<0.65:
                                take = True
                                self.image = cv2.rectangle(self.image, (450,210), (450+380,210+400), (0,255,0), 2)
                            mp_drawing.draw_detection(self.image, detection)
                    #cv2.imshow('MediaPipe Face Detection', self.image)
                #print(self.image.shape)
                # wrong wrong wrong
                if not success:
                    print("空帧.")
                    continue

                # 提高性能
                self.image.flags.writeable = False
                # 转为RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # 镜像
                self.image = cv2.flip(self.image, 1)
                # mediapipe模型处理
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                
                # 判断是否有手掌===================================================
                if results.multi_hand_landmarks:

                    # 遍历每个手掌
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 在画面标注手指
                        print(hand_landmarks)
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                         
                        # 获取手腕根部深度坐标
                        cz0 = hand_landmarks.landmark[0].z
                        for i in range(21): # 遍历该手的21个关键点

                            # 获取3D坐标
                            cx_clic = int(hand_landmarks.landmark[12].x * resize_w)
                            cy_clic = int(hand_landmarks.landmark[12].y * resize_h)
                            cz_clic = hand_landmarks.landmark[12].z
                            depth_z_clic = cz0 - cz_clic
                                                       
                            radius = max(int(100 * (1 + depth_z_clic*5)), 0) # 方块的大小与圆半径成正比
                          
                        # 解析手指，存入各个手指坐标
                        landmark_list = []

                        # 用来存储手掌范围的矩形坐标
                        paw_x_list = []
                        paw_y_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)
                        if landmark_list:
                            # 比例缩放到像素
                            ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)
                            
                            # 设计手掌左上角、右下角坐标
                            paw_left_top_x, paw_right_bottom_x = map(ratio_x_to_pixel,[min(paw_x_list),max(paw_x_list)])
                            paw_left_top_y, paw_right_bottom_y = map(ratio_y_to_pixel,[min(paw_y_list),max(paw_y_list)])

                            # 获取食指指尖坐标
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x =ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y =ratio_y_to_pixel(index_finger_tip[2])

# 获取食指和中指距离
                            # 获取中指指尖坐标
                            middle_finger_tip = landmark_list[12]
                            middle_finger_tip_x =ratio_x_to_pixel(middle_finger_tip[1])
                            middle_finger_tip_y =ratio_y_to_pixel(middle_finger_tip[2])

                            # 中间点
                            between_finger_tip_clic = (middle_finger_tip_x+index_finger_tip_x)//2, (middle_finger_tip_y+index_finger_tip_y)//2
                            # print(middle_finger_tip_x)
                            #middle_finger_point = (middle_finger_tip_x,middle_finger_tip_y)
                            #index_finger_point = (index_finger_tip_x,index_finger_tip_y)
                           
                                                      
# 获取食指和拇指距离
                            # 获取拇指指尖坐标
                            thumb_finger_tip = landmark_list[4]
                            thumb_finger_tip_x =ratio_x_to_pixel(thumb_finger_tip[1])
                            thumb_finger_tip_y =ratio_y_to_pixel(thumb_finger_tip[2])

                            # 中间点
                            between_finger_tip_move = (thumb_finger_tip_x+index_finger_tip_x)//2, (thumb_finger_tip_y+index_finger_tip_y)//2
                            # print(middle_finger_tip_x)
                            #thumb_finger_point = (thumb_finger_tip_x,thumb_finger_tip_y)
                            #index_finger_point = (index_finger_tip_x,index_finger_tip_y)


#===========================================================================================================
                            # 勾股定理计算长度
                            line_len_move = math.hypot((index_finger_tip_x-thumb_finger_tip_x),(index_finger_tip_y-thumb_finger_tip_y))
                            line_len_clic = math.hypot((index_finger_tip_x-middle_finger_tip_x),(index_finger_tip_y-middle_finger_tip_y))
                            # 将指尖距离映射到文字
                            rect_percent_text = math.ceil(line_len_clic)
                            
                            magrin = 60              
                             #===========================================================================================================
                            # 激活模式，需要让矩形跟随移动
                            if squareManager.drag_active_move:
                                # 更新方块
                                squareManager.updateSquare(between_finger_tip_move[0],between_finger_tip_move[1])
                                if(line_len_move>magrin):
                                    # 取消激活
                                    squareManager.drag_active_move =False
                                    squareManager.active_index = -1

                            elif (line_len_move<magrin) and (squareManager.checkOverlay(between_finger_tip_move[0],between_finger_tip_move[1]) != -1 )and( squareManager.drag_active_move  == False):
                                    # 激活
                                    squareManager.drag_active_move =True
                                    # 计算距离
                                    squareManager.setLen(between_finger_tip_move[0],between_finger_tip_move[1])
                
                # 显示方块，传入本实例，主要为了半透明的处理,不改变大小
                #squareManager.display(self,80)
#===========================================================================================================
                            # 激活模式，需要让矩形大小变化
                            if squareManager.drag_active_clic:
                                # 更新方块
                                #squareManager.updateSquare(between_finger_tip_clic[0],between_finger_tip_clic[1])
                                if(line_len_clic>magrin):
                                    # 取消激活
                                    squareManager.drag_active_clic =False
                                    squareManager.active_index = -1

                            elif (line_len_clic<magrin) and (squareManager.checkOverlay(between_finger_tip_clic[0],between_finger_tip_clic[1]) != -1 )and( squareManager.drag_active_clic  == False):
                                    # 激活
                                    squareManager.drag_active_clic =True
                                    # 计算距离
                                    squareManager.setLen(between_finger_tip_clic[0],between_finger_tip_clic[1])

                # 显示方块，传入本实例，主要为了半透明的处理,改变大小
                if squareManager.drag_active_clic: #and not squareManager.drag_active_move:
                    squareManager.display(self,radius)
                    now = time.time()
                    
                    if radius > 130 or bool(eye_count):
                        if (squareManager.active_index == 0 and now-pause>2) or bool(eye_count):
                            '''
                            # 活体检测 live detection
                            if eye_count == 0:
                                eye_count = 1
                            
                            rects = detector(gray, 0)
                            for rect in rects:
                                shape = predictor(gray, rect)
                                shape = face_utils.shape_to_np(shape)
                                # 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
                                leftEye = shape[lStart:lEnd]
                                rightEye = shape[rStart:rEnd]
                                leftEAR = eye_aspect_ratio(leftEye)
                                rightEAR = eye_aspect_ratio(rightEye)
                                EAR = (leftEAR + rightEAR) / 2.0
                                if EAR < eye_margin:
                                    eye_count = eye_count - 1
                                if eye_count == 0:
                                    face=1   
                            '''     
                            face=1          
                            pause=time.time()
                        elif squareManager.active_index == 2 and now-pause>2:
                            #os.system("gnome-terminal -e 'htop'")
                            '''
                            images, labels = load_dataset('./dataset_image/color/'+opt.user)
                            # create 128-Demensions features vector
                            X_embeddings = img_to_encoding(images, facenet) 
                            #print(X_embeddings.shape)
                            features = np.array(X_embeddings)
                            print(features.shape)
                            np.save("./dataset_image/{}_feature".format(opt.user), features)
                            pause=time.time()
                            '''
                        elif squareManager.active_index == 1 and now-pause>3 and take:
                            
                            self.image[:]=255
                            #time.sleep(1)
                            photo=2
                            pause=time.time()
                        elif squareManager.active_index == 1 and now-pause>2 and not take:
                           
                            cv2.putText(self.image, "FOCUS TO CENTER", (430, 200),cv2.FONT_HERSHEY_PLAIN, 3, (35, 51, 233), 6)
                       # elif squareManager.active_index == 3 and now-pause>2:
                    
                    
                    if radius < 110:
                        if squareManager.active_index == 0 :
                            #os.system('ps -ef | pgrep firefox | xargs kill -9')
                            face=0
                            take=False
                        elif squareManager.active_index == 2 :
                            os.system('pgrep htop | xargs kill -s 9')
                    
                        elif squareManager.active_index == 1 and now-pause>2 :
                            #os.system("gnome-terminal -e 'bash -c \"cat body.py | pv -qL 500;exec bash\"'")
                            pause=time.time()
                            #os.system('pgrep -l -n bash | xargs kill -s 9')
                            #os.system('pgrep -l -n pv | xargs kill -s 9')
              
                    
                else:
                    squareManager.display(self,80)
                
             
#===========================================================================================================
                
                
                # lab
                cv2.putText(self.image, "Laboratory", (10, 85),cv2.FONT_HERSHEY_PLAIN, 5, (89, 97, 99), 8)
                if take:
                    cv2.putText(self.image, "OK", (590, 200),cv2.FONT_HERSHEY_PLAIN, 5, (73, 226, 35), 8)
                '''
                if radius > 130:
                    cv2.putText(self.image, "activated:"+str("YES"), (10, 150),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                '''
                '''
                # 显示刷新率FPS
                cTime = time.time()
                fps_text = 1/(cTime-fpsTime)
                fpsTime = cTime
                #cv2.putText(self.image, "FPS: " + str(int(fps_text)), (10, 70),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                '''
                # 显示画面
                # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
                cv2.imshow('LOGIN system', self.image)
                if photo != 0 and take:
                    time.sleep(0.5)
                    photo=photo-1
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()







