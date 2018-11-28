#!usr/bin/python
# -*- coding: utf-8 -*-
#定义编码，中文注释
 
#import the necessary packages
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('C:/Users/Lenovo/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') # 加载人脸特征库
eye_cascade = cv2.CascadeClassifier('C:/Users/Lenovo/Anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')
# 找到目标函数
# 找到目标函数
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0)        
    edged = cv2.Canny(gray, 35, 125)     
    faces = face_cascade.detectMultiScale(image, scaleFactor = 1.15, minNeighbors = 5, minSize = (5, 5)) # 检测人脸  
    for(x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # 用矩形圈出人脸
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        gray = cv2.GaussianBlur(gray, (5, 5), 0)        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 求最大面积 
    a2 = np.ones((5,1,2), dtype=np.int32)
    if x!=0 and y!=0 and (x+w)!=0 and (y+h)!=0: 
        a2[0]=(x,y)
        a2[1]=(x,y+h)
        a2[2]=(x+w,y+h)
        a2[3]=(x+w,y)
        a2[4]=[x,y]
        cnts.append(a2)
        c = cnts[-1]
    else:
        print('cant achieve the point')
    return cv2.minAreaRect(c)

    # compute the bounding box of the of the paper region and return it
    # cv2.minAreaRect() c代表点集，返回rect[0]是最小外接矩形中心点坐标，
    # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
    
 
# 距离计算函数 
def distance_to_camera(knownWidth, focalLength, perWidth):  
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth            
 
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 20.19
 
# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
# A4纸的长和宽(单位:inches)
KNOWN_WIDTH = 6.73
# initialize the list of images that we'll be using
IMAGE_PATHS = ["C:/D/test3.jpeg"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
#读入第一张图，通过已知距离计算相机焦距
image = cv2.imread(IMAGE_PATHS[0]) 
marker = find_marker(image)           
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  
 
#通过摄像头标定获取的像素焦距
#focalLength = 811.82
print('focalLength = ',focalLength)
 #打开摄像头
camera = cv2.VideoCapture(0)
while camera.isOpened():
    # get a frame
    (grabbed, frame) = camera.read()
    marker = find_marker(frame)
    if marker == 0:
        print(marker)
        continue
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    # draw a bounding box around the image and display it
    box = cv2.boxPoints(marker)
    box = np.int0(box)
    for i in range(len(box)):
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    # inches 转换为 cm
            cv2.putText(frame, "%.2fcm" % (inches *30.48/ 12),
                       (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       2.0,(0, 255, 0), 3)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows() 



