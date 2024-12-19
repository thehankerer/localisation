import pandas as pd
import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random
import time

from rotations import Quaternion, skew_symmetric
from vis_tools import *

import cv2
import glob
import os


# To avoid pandas warning
pd.set_option('mode.chained_assignment', None)

# Path to the video file
video_path = "C:\\mygit\\square.mp4"




# Check if the file exists
if not os.path.exists(video_path):
    print("Error: Video file not found.")
    exit()
 
# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
 

def extract_features(image):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.05, nlevels=12, edgeThreshold=31, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31, fastThreshold=1 )
    #orb = cv2.ORB_create(nfeatures = 500)
    kp, des = orb.detectAndCompute(image, None)
    return kp,des

def match_features(des1,des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    #match = bf.match(des1,des2)
    match = bf.knnMatch(des1,des2, k = 2)
    return match

def filter_matches_distance(match, dist_threshold):
    filtered_match = []
    
    while True:
        filtered_match = []
        for m,n in match :
            if m.distance < dist_threshold * n.distance:
                filtered_match.append([m])
                
        if len(filtered_match) > 4:
            break
        
        dist_threshold += 0.005
    return filtered_match
   
    
def visualize_matches(image1,kp1,image2,kp2,match):
    match_img = cv2.drawMatchesKnn(image1,kp1,image2,kp2,match, None,flags=2)
    #match_img = cv2.drawMatches(old_gray, kp1, frame_gray, kp2, match[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Feature Tracking with BFMatcher', match_img)
    
    
    
def estimate_motion(match,kp1,kp2,k,depth1=None):
    rmat = np.eye(3)#identity matrix
    tvec = np.zeros((3,1))
    image1_points = []
    image2_points = []
    objectpoints = [] #3D point in the camera frame
    
    for m in match:
        m = m[0]
        
        query_idx = m.queryIdx #This is the first image indexes
        # Index of the descriptor and feature in the second image
        train_idx = m.trainIdx #This is the second image indexes
        
        # get first img 3d data (with depth)
        p1_x, p1_y = kp1[query_idx].pt
        image1_points.append([p1_x, p1_y])
    
        # get second img 2d data
        p2_x, p2_y = kp2[train_idx].pt
        image2_points.append([p2_x, p2_y])
        
        p1_z = 3.0
        
        # Transform points from image coordinate frame to camera frame
        scaled_point = np.dot(np.linalg.inv(k), np.array([p1_x, p1_y, 1]))
        # Map the homogeneous coordinates to cartesian
        p1_3d = scaled_point * p1_z
        # Save the camera 3d point
        objectpoints.append(p1_3d)
        
    # Transpose 3D points to feed them into pnp        
    objectpoints = np.array(objectpoints)
    # 2D Points of the second image
    imagepoints = np.array(image2_points)
    #distcoeff = np.array([[0.20532272, -0.35258025,  0.00301216, -0.00463005, -0.15557039]], dtype=np.float32)
    distcoeff = np.array([[0.1899950572, -0.740542480,  -0.000606893304, 0.000224588063, 0.795872765]], dtype=np.float32)
    
    # Solve PnP Ransac
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, distCoeffs = None, flags=cv2.SOLVEPNP_ITERATIVE, 
                                          iterationsCount = 10000, reprojectionError = 0.01, confidence=0.999)
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points

def estimate_trajectory(estimate_motion, matches, kp1, kp2, k , loop,old_gray, frame_gray, P = None):
    # Final trajectory
    #trajectory = [[]]
    location = [np.array([0, 0, 0])]
    # Save timestamps
    timestamp = []
    # Here P is equal to T_nc at first time step
    P_i = np.eye(4)
    # Variable to store Quaternions
    Rvecs = np.zeros([len(matches), 4])
    # Maximum allowed traslation between two consecutive image frames
    max_allowed = 1
    
        #Extract relevant information of the matches, and features from image i to i + 1
    rmat, tvec, image1_points, image2_points = estimate_motion(matches, kp1, kp2, k, depth1 = 3.0)
    #image_move = visualize_camera_movement(old_gray,image1_points,frame_gray, image2_points)
    #cv2.imshow('Motion estimation',image_move)
    R = rmat
    t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
        
        
        # Invert transformation matrix to point from frame i to  i+1
    P_new = np.eye(4) # New P Matrix
    P_new[0:3,0:3] = R.T 
    P_new[0:3,3] = (-R.T).dot(t) 
        # Stack transformation matrices only if the traslation is valid
    if np.abs(t.max()) < max_allowed and np.abs(t.min()) < max_allowed:
        P = P.dot(P_new) # Dot product to stack the matrices
        location = P[:3,3]
            # Save trajectory
        #trajectory.append(P[:3,3])
            # Save the current rotation
        rvec, _ = cv2.Rodrigues(P[0:3,0:3])
        #Rvecs[i] = Quaternion(axis_angle = rvec).to_numpy()
        
        Rvecs = Quaternion(axis_angle = rvec).to_numpy()
        location = np.array(location).T
        #trajectory = np.array(trajectory).T
        Rvecs = Rvecs.T
            # Save time-stamp
        #timestamp.append()
    else:
        print("Too high traslation at time index {0}, Max value is: {1}, Min value is: {2}\n".format(loop, t.max(), t.min()))
        location = np.array([0,0,0])    
    # Final trajectory matrix
    

    return location, Rvecs,P 
    
    

# Read the first frame
finaltrajectory = [np.array([0, 0, 0])]
Rvecsfinal = [np.array([0,0,0,0])]
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame. Video may be corrupted or unsupported.")
    exit()

# Convert to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

kp1, des1 = extract_features(old_gray)

Precent = np.array([[0, -1, 0, 0],
                 [-1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1]], dtype=np.float32)
# Create BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

k = np.array([[3231, 0, 2049],
                [0, 3232, 1543],
                [0,   0,   1]], dtype=np.float32)
loop = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = extract_features(frame_gray)
    print("Number of features detected in frame {0}: {1}\n".format(loop, len(kp2)))

    if des1 is not None and des2 is not None:
        matches = match_features(des1,des2)
        dist_threshold = 0.001
        filtered_match = filter_matches_distance(matches,dist_threshold)
        print("Number of features matched in frames {0} after filtering by distance: {1}".format(loop, len(filtered_match)))
        visualize_matches(old_gray,kp1,frame_gray,kp2,filtered_match)
        
        #rmat, tvec, image1_points, image2_points = estimate_motion(filtered_match, kp1, kp2, k, depth1=3.0)
        
        #image_move = visualize_camera_movement(old_gray,image1_points,frame_gray, image2_points)
        #cv2.imshow('Motion estimation',image_move)
        
        trajectory, Rvecs,Precent = estimate_trajectory(estimate_motion, filtered_match, kp1, kp2, k, loop,old_gray, frame_gray,Precent)
        if np.any(trajectory != 0) or loop ==0 :
            finaltrajectory.append(trajectory)
        
        if np.any(Rvecs != 0) or loop ==0 :
            Rvecsfinal.append(Rvecs)
        #finaltrajectory_np = np.array(finaltrajectory)
        #print(finaltrajectory[-1][0])
        #print(finaltrajectory[-1][1])
        #print(finaltrajectory[-1][2])
        '''
        x_coords = [point[0] for point in finaltrajectory]
        y_coords = [point[1] for point in finaltrajectory]
        z_coords = [point[2] for point in finaltrajectory]
        
        print(x_coords, y_coords, z_coords)
         '''      
        #plt.figure(figsize = (16,12), dpi = 100)
        #plt.imshow(image_move)
        #plt.show()
        
        
        # Update keypoints and descriptors
        old_gray = frame_gray.copy()
        kp1, des1 = kp2, des2


    loop +=1
    # Break loop on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    

print(finaltrajectory)
print(Rvecsfinal)
visualize_trajectory(finaltrajectory, 'Estimated Trajectory with VO')
visualize_angles(Rvecsfinal, 'Estimated Rotations with VO')
cap.release()

#cv2.destroyAllWindows() 

 