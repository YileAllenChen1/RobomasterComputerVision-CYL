import cv2 as cv
import math
import numpy as np 

cap = cv.VideoCapture('media/runeDemo.mp4')

stateVal = 4 # number of states: coordinates and velocity (x,y,dx,dy)
measureVal = 2 # measurement value: number of coordinates observed

KF = cv.KalmanFilter(stateVal, measureVal,0)

#measurement = np.zeros((measureVal, 1))
"""
KF.transitionMatrix = np.array([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], np.float32) # state transition matrix A with dimension (stateVal, stateVal)

# not set control matrix B, default 0
KF.measurementMatrix = np.zeros((measureVal, stateVal), np.float32)
np.fill_diagonal(KF.measurementMatrix,1.)    # measurement matrix 
KF.processNoiseCov = 9.9999997e-06 * np.identity(stateVal, np.float32)  # Q: Process noise covariance matrix, unit matrix
KF.measurementNoiseCov = 1e-1 * np.identity(measureVal, np.float32)     # R: measurement noise covariance matrix, unit matrix
KF.errorCovPost = 1 * np.identity(stateVal, np.float32)                 # P: posteriori Error estimate covariance matrix, unit matrix
"""
measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)
#KF.statePost = 0.1 * np.random.randn(stateVal,1)            # initialize to random value
KF.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
KF.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)   # state transition matrix A with dimension (stateVal, stateVal)
KF.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 9.9999997e-06 # Q: Process noise covariance matrix, unit matrix
KF.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1 # R: measurement noise covariance matrix, unit matrix
KF.errorCovPost = np.identity(stateVal, np.float32) * 1 # P: posteriori Error estimate covariance matrix, unit matrix

state = 0.1 * np.random.randn(stateVal, measureVal)

while True:
    ret,frame = cap.read()
    binary = frame
    frame = cv.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))    #frame = cv.resize(frame, None, fx=0.5,fy=0.5)
    binary = cv.resize(binary, (int(binary.shape[1] * 0.5), int(binary.shape[0] * 0.5)))    #binary = cv.resize(binary, None, fx=0.5,fy=0.5)
       
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, frame= cv.threshold(frame, 80, 255, cv.THRESH_BINARY) # need to modify threshold 80
    
    kernel = np.ones((5,5),np.uint8)
    frame = cv.dilate(frame,kernel,iterations = 1)

    mask = np.zeros((frame.shape[0]+2,frame.shape[1]+2),np.uint8)
    cv.floodFill(frame, mask, (5,50), (255,0), cv.FLOODFILL_FIXED_RANGE)

    _, frame = cv.threshold(frame, 80, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # RETR_LIST, CHAIN_APPROX_NONE
    
    for contour in contours:
        # ignore contours smaller/larger than threshold
        if cv.contourArea(contour) < 50 or cv.contourArea(contour) > 1e4:
            continue
        cv.drawContours(frame, contours, -1, (255, 0, 0), 2) # first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        rrect = cv.fitEllipse(contour)  # rrect: ((center_coordinates), (axesLength), angle)
        height = rrect[1][1]
        width = rrect[1][0]
        aim = height/width  # get approximate ratio of armour board
        
        box = cv.boxPoints(rrect)   # get the four vertices of the rotated rectanglce
        box = np.int0(box)          # convert to int
        
        if(aim > 1.7 and aim < 2.6):    # ratio range for armor plates
            cv.drawContours(binary,[box], -1, (0, 255, 255), 4)    # draw contour on target armor plate
            #cv.imshow('frmae feed', frame)
            mid = 100000.0
            for i in range(1,len(contours)): # check the rest of the contours
                area = cv.contourArea(contours[i])
                if(area < 50 or area > 1e4):    
                    continue
                rrectA = cv.fitEllipse(contours[i])
                heightA = rrectA[1][1]
                widthA = rrectA[1][0]
                aimA = heightA/widthA
                if aimA > 3.0:  # ratio range for fan blade
                    boxA = cv.boxPoints(rrectA)   # get the four vertices of the rotated rectanglce
                    boxA = np.int0(boxA)
                    cv.drawContours(binary,[boxA], -1, (0, 128, 255), 4)
                    
                    M = cv.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    Ma = cv.moments(contours[i])
                    cXa = int(Ma["m10"] / Ma["m00"])
                    cYa = int(Ma["m01"] / Ma["m00"])

                    distance = math.sqrt((cX-cXa)**2 + (cY-cYa)**2)
                    
                    if(mid > distance):
                        mid = distance
            if mid > 60:     # this value needs to be tuned based on actual situation, related to image dimension and object distance
                cv.circle(binary, (int(rrect[0][0]), int(rrect[0][1])), 13, (0,0,255),2)
                #cv.ellipse(binary, (int(rrect[0][0]), int(rrect[0][1])), (int(rrect[1][0]), int(rrect[1][1])), 0,0,360,(0,0,255),2)
                #cv.drawContours(binary, [box], 0, (0, 0, 255), 1)
                
                prediction = KF.predict()
                predict_point = (prediction[0], prediction[1])
                #print("h ", type(prediction))
                #measurement[0] = rrect[0][0]
                #measurement[1] = rrect[0][1]
                #measurement= np.reshape(measurement, (1,2))
                #measurement = np.array([[int(rrect[0][0])],[int(rrect[0][1])]])
                measurement = np.array([[np.float32(rrect[0][0])],[np.float32(rrect[0][1])]])
                #print(measurement.shape)
                #measurement = KF.measurementNoiseCov * np.random.randn(1, 1)
                #print("1", measurement)
                # generate measurement
                #measurement = np.dot(KF.measurementMatrix, state) + measurement
                #print("2", measurement)
                KF.correct(measurement)
                
                cv.circle(binary, predict_point, 3, (34, 255, 255), -1) #prediction circle
                
                temp = []
                tempList=[]
                #print("rrect ", rrect)
                for i in rrect[:2]:
                    #print("i ",i)
                    for j in i:
                        #print("j ",j)
                        tempList.append(j)
                   # print("temlist ", tempList)
                    temp.append(tempList)
                    tempList = []
                temp.append([rrect[-1]])
                #temp = [list(i) for i in rrect]  
                #temp = map(list, rrect)
                #print(type(temp))
                #print(temp)
                #print("type predict ", type(prediction[0][0]))
                temp[0][0] = prediction[0][0]
                temp[0][1] = prediction[1][0]
                #print("updated  ",temp)
                rrectTupe=()
                rrect = ()
                """
                for i in temp[:2]:
                    print("i ",i)
                    for j in i:
                        print("j ",j)
                        rrectTupe.append(j)
                    #print(i,j)
                    rrect.append(rrectTupe)
                    rrectTupe = ()
                rrect.append((temp[-1]))
                """
                rrect = tuple(temp)
                #rrect = [tuple(i) for i in temp] 
                #print("rrect after ", rrect)
                #prin
                #rrect[0][0] = prediction[0]
                #rrect[0][1] = prediction[1]
              
    cv.imshow('Video feed', frame)
    cv.imshow('binary feed', binary)
    cv.waitKey(1)

"""
An ellipse is defined by 5 parameters:

xc : x coordinate of the center
yc : y coordinate of the center
a : major semi-axis
b : minor semi-axis
theta : rotation angle

RotatedRect e = fitEllipse(points);
return: center_coordinates, axesLength, angle

float xc    = e.center.x;
float yc    = e.center.y;
float a     = e.size.width  / 2;    // width >= height
float b     = e.size.height / 2;
float theta = e.angle;              // in degrees
"""

"""
[9.9999997e-06, 0, 0, 0;
 0, 9.9999997e-06, 0, 0;
 0, 0, 9.9999997e-06, 0;
 0, 0, 0, 9.9999997e-06]
"""
"""
[0.1, 0;
 0, 0.1]
"""
"""
[1, 0, 0, 0;
 0, 1, 0, 0;
 0, 0, 1, 0;
 0, 0, 0, 1]
"""
"""
[-1.6030947e-10;
 0.015813505;
 -0.070092797;
 -0.042859599]
"""