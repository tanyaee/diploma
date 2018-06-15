import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

'''program execution time'''
start_time = time.clock()

original_image = cv2.imread('image/industrial.jpg', 0)
surf = cv2.xfeatures2d.SURF_create(5000, extended=True,upright=True)
kp, des = surf.detectAndCompute(original_image, None)
img_kp = cv2.drawKeypoints(original_image,kp,None,(255,0,0),4)
print('number of key points of original image:', len(des))
plt.imshow(img_kp),plt.show()
print("descriptor size", surf.descriptorSize())

'''image cropping'''
I1 = original_image[0:400, 0:400]

'''image to double representation'''
image_1 = cv2.normalize(I1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
'''transform matrix to column vector '''
A1 = image_1.flatten(order='F')

'''covariance calculation'''
image1_covariance = np.cov(A1)
#print("covariance", c1)

'''crop second image'''
I2 = original_image[0:500, 0:500]
image_2 = cv2.normalize(I2.astype('float'),None,0.0,1.0, cv2.NORM_MINMAX)
A2 = image_2.flatten(order='F')

image2_covariance = np.cov(A2)
covariance = image1_covariance/image1_covariance
image2_variance = np.var(image_2)
print("variance of second image: ", image2_variance)

'''signal to noise ratio calculation'''
snr2= np.mean(image_2)/image2_variance
print("snr", snr2)

kp2, des2 = surf.detectAndCompute(I2, None)

print(time.clock() - start_time, "seconds")
cv2.waitKey(0)

'''Invormativity index calculation'''
H = image2_variance + snr2 + len(des)-covariance
print("informativity index", H)
