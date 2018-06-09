import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('image/template.jpg',0)
img2 = cv2.imread('image/picture.jpg',1)

# Initiate ORB detector
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher()
# Match descriptors.
matches = bf.match(des1,des2)
dist = [m.distance for m in matches]
thres_dist = (sum(dist) / len(dist)) * 0.5
matches = [m for m in matches if m.distance < thres_dist]
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
distance1 = []
for m in matches:
    distance = m.distance
    if distance < 0.75:
        distance1.append(m)

# Draw first 10 matches.
'''raw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
'''

img3 = np.zeros((1,1))
im4 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:500],None, flags=2)
#im4 = cv2.drawMatches( img1,kp1,img2,kp2,matches,distance1,flags=2)
plt.imshow(im4),plt.show()
print (len(kp1))
print (len(kp2))
print('matches number', len(distance1))