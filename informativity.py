import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
#from scipy import ndimage

'''program execution time'''
start_time = time.clock()

img = cv2.imread('image/industrial.jpg',0)
surf = cv2.xfeatures2d.SURF_create(5000, extended=True,upright=True)
kp, des = surf.detectAndCompute(img, None)
img_kp = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
print('number of key points of img:', len(des))
plt.imshow(img_kp),plt.show()
print("descriptor size", surf.descriptorSize())

'''image cropping'''
I1 = img[0:400, 0:400]

#cv2.imshow("cropped", I1)#,plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''image to double representation'''
img1 = cv2.normalize(I1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

'''covariance calculation'''
c1 = np.cov(img1)
#print("covariance", c1)

'''crop second image'''
I2 = img[0:500, 0:500]
img2 = cv2.normalize(I2.astype('float'),None,0.0,1.0, cv2.NORM_MINMAX)
c2 = np.cov(img2)
#c3 = c2/c1
v2 = np.var(img2)
print("variance img2: ", v2)
'''SNR calculation'''
snr2= np.mean(img2)/v2
print("snr", snr2)

kp2, des2 = surf.detectAndCompute(I2, None)

print(time.clock() - start_time, "seconds")
cv2.waitKey(0)

'''Invormativity index'''
H = v2 + snr2 + len(des)
print("informativity index", H)
