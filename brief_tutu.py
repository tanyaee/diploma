import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

img1 = cv2.imread('image/industrial.jpg', 0)
img2 = cv2.imread('image/industrial_scale.jpg', 0)

# Initiate ORB detector
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#bf = cv2.BFMatcher()
# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
distance1 = []
sumDistances = 0
for m in matches:
    sumDistances += m.distance
    #print(m.distance)
avgDistance = sumDistances / len(matches)
print(avgDistance)
for m in matches:
    if m.distance < avgDistance:
       distance1.append(m)

img3 = np.zeros((1,1))
im4 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:200],None, flags=2)
#im4 = cv2.drawMatches( img1,kp1,img2,kp2, img3,None flags=2)
plt.imshow(im4),plt.show()
print ('img1:',len(des1))
print ('img2:',len(des2))
print('matches number', len(distance1))
start_time = time.clock()

print (time.clock() - start_time, "seconds")