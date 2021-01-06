import cv2 as cv
import math
import numpy as np 

cap = cv.VideoCapture('media/Demo.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
#cv.VideoWriter_fourcc(*'MP4V'),0x00000021
#result = cv.VideoWriter('results.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, size) 

stateVal = 4 # number of states: coordinates and velocity (x,y,dx,dy)
measureVal = 2 # measurement value: number of coordinates observed

KF = cv.KalmanFilter(stateVal, measureVal, 0)

measurement = np.zeros((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)
#KF.statePost = 0.1 * np.random.randn(stateVal,1)            # initialize to random value
KF.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
KF.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)   # state transition matrix A with dimension (stateVal, stateVal)
KF.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 9.9999997e-06 # Q: Process noise covariance matrix, unit matrix
KF.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1 # R: measurement noise covariance matrix, unit matrix
KF.errorCovPost = np.identity(stateVal, np.float32) * 1 # P: posteriori Error estimate covariance matrix, unit matrix

state = 0.1 * np.random.randn(stateVal, measureVal)
state = np.array(state, np.float32)

while True:
    ret,frame = cap.read()
    binary = frame
    #frame = cv.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))    #frame = cv.resize(frame, None, fx=0.5,fy=0.5)
    #binary = cv.resize(binary, (int(binary.shape[1] * 0.5), int(binary.shape[0] * 0.5)))    #binary = cv.resize(binary, None, fx=0.5,fy=0.5)
    #print(frame.shape[1] * 0.5, frame.shape[0] * 0.5)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    ret, frame= cv.threshold(frame, 80, 255, cv.THRESH_BINARY) # need to modify threshold 80
    """
    kernel = np.ones((5,5),np.uint8)
    frame = cv.dilate(frame,kernel,iterations = 1)
    
    mask = np.zeros((frame.shape[0]+2,frame.shape[1]+2),np.uint8)
    cv.floodFill(frame, mask, (5,50), (255,0), cv.FLOODFILL_FIXED_RANGE)
    
    _, frame = cv.threshold(frame, 80, 255, cv.THRESH_BINARY_INV)
    """
    contours, hierarchy = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # RETR_LIST, CHAIN_APPROX_NONE
    
    hi = 0
    heights = {}
    R = {} 
    RA = {}

    for contour in contours:
        # ignore contours smaller/larger than threshold
        if cv.contourArea(contour) < 20 or cv.contourArea(contour) > 1e3:
            continue
        cv.drawContours(frame, contours, -1, (255, 0, 0), 2) 
        rrect = cv.fitEllipse(contour)  # rrect: ((center_coordinates), (axesLength), angle)
        height = rrect[1][1]
        width = rrect[1][0]
        
        box = cv.boxPoints(rrect)   # get the four vertices of the rotated rectanglce
        box = np.int0(box)          # convert to int
        
        #for i in range(4):

        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])   
        cY = int(M["m01"] / M["m00"])
        #cX = rrect[0][0]
        #cY = rrect[0][1]
        for i in range(1,len(contours)): # check the rest of the contours, generally 2 in this case
            area = cv.contourArea(contours[i])
            if(area < 20 or area > 1e3):    
                continue
            rrectA = cv.fitEllipse(contours[i])
            #print(rrectA)
            boxA = cv.boxPoints(rrectA)   # get the four vertices of the rotated rectanglce
            boxA = np.int0(boxA)

            #cv.drawContours(binary,[box], -1, (0, 255, 255), 4)
            #cv.drawContours(binary,[boxA], -1, (0, 255, 0), 4)

            Ma = cv.moments(contours[i])
            cXa = int(Ma["m10"] / Ma["m00"])
            cYa = int(Ma["m01"] / Ma["m00"])
            #cXa = rrectA[0][0]
            #cYa = rrectA[0][1]

            slop = abs(rrect[2]-rrectA[2])     # angle difference
            heightA = rrectA[1][1]
            widthA = rrectA[1][0]
            distance = math.sqrt((cX-cXa)**2 + (cY-cYa)**2)

            if height > heightA:
                max_height = height
                min_height = heightA
            else:
                max_height = heightA
                min_height = height

            line_x = abs(cX-cXa)
            difference = max_height - min_height
            aim = distance/((height+heightA)/2)
            difference3 = abs(width-widthA)
            height_total = (height+heightA)/200
            slop_low = abs(rrect[2]+rrectA[2])/2

            
            if ((aim < 3.0 - height_total and 
                    aim > 2.0 - height_total and 
                    slop <=5 and 
                    difference <=8 and 
                    difference3 <=5 and 
                    (slop_low <=30 or slop_low>=150) and 
                    line_x>0.6*distance) or 
                    (aim < 5.0-height_total and 
                    aim > 3.2 - height_total or 
                    slop <= 7 and 
                    difference <=15 and 
                    difference3 <= 8 and 
                    (slop_low <= 30 or slop_low >=150) and 
                    line_x >0.7*distance)):                                                                                                                                                                                                                       
                heights[hi] = (height + heightA) / 2
                
                R[hi] = rrect
                RA[hi] = rrectA
                print("rrect ",rrect, "rrectA ",rrectA)
                print("-----------------------------------------")
                print(hi, R[hi], RA[hi])
                hi+=1
                cv.drawContours(binary,[box], -1, (0, 255, 255), 4)
                cv.drawContours(binary,[boxA], -1, (0, 255, 0), 4)

    max = 0.0                                                     
    mark = 0
    #print(len(R), len(RA))
    for i in range(hi):     # multiple targets, hit the nearest
        if heights[i] >= max:
            max = heights[i]
            mark = i
    #print(hi)
    #print(R[mark], RA[mark])
    if hi != 0:
        cv.circle(binary, (int(R[mark][0][0]+RA[mark][0][0])/2, 
                            int(R[mark][0][1]+RA[mark][0][1])/2), 15, (0,0,255),4) # circle the target armor to hit

        center_x = int(R[mark][0][0]+RA[mark][0][0])/2
        center_y = int(R[mark][0][1]+RA[mark][0][1])/2

        prediction = KF.predict()
        predict_point = (prediction[0], prediction[1])
        #measurement = np.array([[np.float32(rrect[0][0])],[np.float32(rrect[0][1])]]) # center coordinate of rrect
        measurement = np.array([[np.float32(center_x)],[np.float32(center_y)]])
        KF.correct(measurement)
        
        cv.circle(binary, predict_point, 3, (34, 255, 255), -1) #prediction circle

        center_x = prediction[0]
        center_y = prediction[1]
        #temp = ((predict_point[0][0], predict_point[1][0]), rrect[1], rrect[2])
        #rrect = temp




              
    cv.imshow('Video feed', frame)
    cv.imshow('binary feed', binary)
    cv.waitKey(1)
    #result.write(binary)
cap.release() 
#result.release() 

"""
An ellipse is defined by 5 parameters:

xc : x coordinate of the center
yc : y coordinate of the center
a : major semi-axis
b : minor semi-axis
theta : rotation angle

RotatedRect e = fitEllipse(points);
return: center_coordinates, axesLength, angle
"""