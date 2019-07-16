#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:53:06 2019

@author: jeffreymu jmu22@bu.edu
partner : dojun park  parkdj@bu.edu
"""

import numpy as np
import cv2



class gestures():

    bg = None
    
    def __init__(self):
        #constructor
        self.background = gestures.bg
        
        
    # Function to determine skin method #2
    ### USE THIS ONE

    def SkinDetect2(self,frame):
        """Used to detect skin from video stream; play with HSV values to get better threshold"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        UpperSkin = np.array([30, 0.85*255, 0.8*255])
        LowerSkin = np.array([0, 0.2*255, 0.05*255])
        
        #Detect skin using HSV bounds
        mask = cv2.inRange(hsv, LowerSkin, UpperSkin)
        return mask

    # Function that accumulates the frame differences for a certain number of pairs of frames
    # mh - vector of frame difference images
    # dst - the destination grayscale image to store the accumulation of the frame difference images
    def myMotionEnergy(self,mh):
        # the window of time is 3
        mh0 = mh[0]
        mh1 = mh[1]
        mh2 = mh[2]
        dst = np.zeros((mh0.shape[0], mh0.shape[1], 1), dtype = "uint8")
        for i in range(mh0.shape[0]):
            for j in range(mh0.shape[1]):
                if mh0[i,j] == 255 or mh1[i,j] == 255 or mh2[i,j] == 255:
                    dst[i,j] = 255
        return dst
    
    
    def ravg(self, frame, weight):
        #computes the running average
        if gestures.bg is None:
            #if no background set it 
            gestures.bg = frame.copy().astype("float")
            return
       
        cv2.accumulateWeighted(frame, gestures.bg,weight)
        
            
    def frameDiff(self,frame):
        #frame differencing
        
        
        #absolute difference 
        absdiff = cv2.absdiff(gestures.bg.astype("uint8"), frame)
        
        _, threshold = cv2.threshold(absdiff, 10, 255, cv2.THRESH_BINARY)
        return threshold

        
    def resize(self,frame, width, height):
        h, w = frame.shape[:2]
      
        
        #keeps the aspect ratio
        r = width / float(w)
        dim = (width, int(h * r))
        resz = cv2.resize(frame, dim)

        return resz


if __name__ == "__main__":
    g = gestures()
    #uses this in the gestures class


    #reads all the template images in grayscale
    
    thumbsup = cv2.imread("thumbsup.png",cv2.IMREAD_GRAYSCALE)
    peace = cv2.imread("peace.png", cv2.IMREAD_GRAYSCALE)
    rock = cv2.imread("rock.png", cv2.IMREAD_GRAYSCALE)

    #sets a weight for the running average
    
    weight = 0.5
    cap = cv2.VideoCapture(0)

    
    num_frames = 0
    
    #resizes the templates to be smaller that the frame
    thumbsup = g.resize(thumbsup, width=100, height = None)
    peace = g.resize(peace, width=100, height = None)
    rock = g.resize(rock, width=100, height = None)
    
    #gets the positions of the templates
    tw,th = thumbsup.shape[::-1]
    pw,ph = peace.shape[::-1]
    rw,rh = rock.shape[::-1]
    
    while(True):
        
        
        #read each frame
        _, prev_frame = cap.read()
        
        
        #resize each frame to a certain height
        prev_frame = g.resize(prev_frame, width=800, height = None)
        
        #flip it because its a mirror image
        
        prev_frame = cv2.flip(prev_frame,1)
      
    
        g_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        
        
        
        g_frame = cv2.GaussianBlur(g_frame, (7,7), 0)
    
        if num_frames < 10:
            #takes the average of the first 10 frames

            g.ravg(g_frame, weight)
            
        else:
            
            
            
             #skin detect algorithim, works better without it but still works with it 
            
#            skin = g.SkinDetect2(prev_frame)
#            res = cv2.bitwise_and(prev_frame,prev_frame, mask= skin)
#
#            ret, threshed_img = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY),
#            10, 255, cv2.THRESH_BINARY)
#            cv2.imshow('Skin Detection', res)
#            
#            contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)
#            c = max(contours, key = cv2.contourArea)
#            a, b, c, d = cv2.boundingRect(c)
#            handrect = cv2.rectangle(prev_frame, (a, b), (a+c, b+d), (0, 255, 0), 3)
#
            threshold = g.frameDiff(g_frame)

            if threshold is not None:  
               
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                #matches the thresholded image to the templates
                result1 = cv2.matchTemplate(threshold, thumbsup, cv2.TM_CCOEFF_NORMED)
                result2 = cv2.matchTemplate(threshold, peace, cv2.TM_CCOEFF_NORMED)
                result3=  cv2.matchTemplate(threshold, rock, cv2.TM_CCOEFF_NORMED)
    
    
                #where the matching is at least 0.6
                
                loc1 = np.where(result1 >= 0.6)
                loc2 = np.where(result2 >=0.6)
                loc3 = np.where(result3 >= 0.6)
                
                
                #iterate through each of the locs and make a colored rectangle for each one
                for pt in zip(*loc1[::-1]):
                 
                        cv2.rectangle(prev_frame, pt, (pt[0]+tw, pt[1]+th),(0,255,0),2)
                        
                        cv2.putText(prev_frame,'thumbs up!',(tw,th), font, 1,(255,255,255),2,cv2.LINE_AA)
            
                for pt in zip(*loc2[::-1]):
                 
                        cv2.rectangle(prev_frame, pt, (pt[0]+pw, pt[1]+ph),(0,0,255),2)
                        cv2.putText(prev_frame,'peace',(pw,ph), font, 1,(255,255,255),2,cv2.LINE_AA)

                for pt in zip(*loc3[::-1]):
                 
                        cv2.rectangle(prev_frame, pt, (pt[0]+rw, pt[1]+rh),(255,0,0),2)
                        cv2.putText(prev_frame,'rock!',(rw,rh), font, 1,(255,255,255),2,cv2.LINE_AA)
                    
                        
                                    
        #increment frames
        num_frames += 1
        
        cv2.imshow("Frame", prev_frame)

        keypress = cv2.waitKey(1)
        
        if keypress == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
            
    
        
        

