import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img1 = cv2.imread('image/industrial.jpg', 0)
img2 = cv2.imread('image/industrial_scale.jpg', 0)

surf = cv2.xfeatures2d.SURF_create(1000)
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
# create BFMatcher object
#bf = cv2.BFMatcher()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.

matches = bf.match(des1,des2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
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
        
# Draw first n matches.
img3 = np.zeros((1,1))

im4 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:200], img3, flags=2)
plt.imshow(im4)
cv2.imwrite('img4.jpg',im4)
print ('img1:',len(des1))
print ('img2:',len(des2))
print('matches number', len(distance1))

'''program execution time'''
start_time = time.clock()

print (time.clock() - start_time, "seconds")
