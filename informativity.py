import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
#from scipy import ndimage


img = cv2.imread('image/industrial.jpg',0)
'''image cropping'''
I1 = img[0:400, 0:400]
#cv2.imshow("cropped", crop_img)#,plt.show()
#cv2.waitKey(0)
cv2.destroyAllWindows()
'''image to double representation'''
img1 = cv2.normalize(I1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#print(I1)
'''covariance calculation'''
c1 = np.cov(img1)
print("covariance", c1)
I2 = img[0:500, 0:500]
img2 = cv2.normalize(I2.astype('float'),None,0.0,1.0, cv2.NORM_MINMAX)
c2 = np.cov(img2)
#c3 = c2/c1



surf = cv2.xfeatures2d.SURF_create(5000)
kp, des = surf.detectAndCompute(I1, None)
img2 = cv2.drawKeypoints(I1,kp,None,(255,0,0),4)
print('img1:', len(des))
plt.imshow(img2),plt.show()
print(surf.descriptorSize())
'''program execution time'''
start_time = time.clock()

print(time.clock() - start_time, "seconds")
#image -= img.mean()
cv2.waitKey(0)