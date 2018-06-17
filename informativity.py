import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

'''program execution time'''
start_time = time.clock()

original_image = cv2.imread('image/industrial.jpg', 0)
height, width = original_image.shape[:2]
print(height,width)
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


print(time.clock() - start_time, "seconds")
cv2.waitKey(0)


for i in range(1, 6):
     if i == 1:
        '''crop second image'''
        I2 = original_image[0:500, 0:500]
        image_2 = cv2.normalize(I2.astype('float'),None,0.0,1.0, cv2.NORM_MINMAX)
        A2 = image_2.flatten(order='F')

        image2_covariance = np.cov(A2)
        covariance = image2_covariance/image1_covariance
        variance = np.var(image_2)
        print("variance of second image: ", variance)

        '''signal to noise ratio calculation'''
        snr= np.mean(image_2)/variance
        print("snr", snr)

        kp, des = surf.detectAndCompute(I2, None)
        H1 = variance + snr + len(des) - covariance
        print("H1", H1)
     if i == 2:
        '''crop second image'''
        I2 = original_image[0:700, 0:700]
        image_2 = cv2.normalize(I2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        A2 = image_2.flatten(order='F')

        image2_covariance = np.cov(A2)
        covariance = image2_covariance/image1_covariance
        variance = np.var(image_2)
        print("variance of third image: ", variance)

        '''signal to noise ratio calculation'''
        snr= np.mean(image_2)/variance
        print("snr 3", snr)

        kp, des = surf.detectAndCompute(I2, None)
        '''Invormativity index calculation'''
        H2 = variance + snr + len(des) - covariance
        print("H2", H2)
     if i == 3:
        '''crop second image'''
        I2 = original_image[0:900, 0:900]
        image_2 = cv2.normalize(I2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        A2 = image_2.flatten(order='F')

        image2_covariance = np.cov(A2)
        covariance = image2_covariance/image1_covariance
        variance = np.var(image_2)
        print("variance of third image: ", variance)

        '''signal to noise ratio calculation'''
        snr= np.mean(image_2)/variance
        print("snr 3", snr)

        kp, des = surf.detectAndCompute(I2, None)
        '''Invormativity index calculation'''
        H3 = variance + snr + len(des) - covariance
        print("H3", H3)
     if i == 4:
        '''crop second image'''
        I2 = original_image[0:1100, 0:1100]
        image_2 = cv2.normalize(I2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        A2 = image_2.flatten(order='F')

        image2_covariance = np.cov(A2)
        covariance = image2_covariance / image1_covariance
        variance = np.var(image_2)
        print("variance of third image: ", variance)

        '''signal to noise ratio calculation'''
        snr = np.mean(image_2) / variance
        print("snr 3", snr)

        kp, des = surf.detectAndCompute(I2, None)
        '''Invormativity index calculation'''
        H4 = variance + snr + len(des) - covariance
        print("H4", H4)

     if i == 5:
        '''crop second image'''
        I2 = original_image[0:1300, 0:1300]
        image_2 = cv2.normalize(I2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        A2 = image_2.flatten(order='F')

        image2_covariance = np.cov(A2)
        covariance = image2_covariance/image1_covariance
        variance = np.var(image_2)
        print("variance of third image: ", variance)

        '''signal to noise ratio calculation'''
        snr= np.mean(image_2)/variance
        print("snr 3", snr)

        kp, des = surf.detectAndCompute(I2, None)
        '''Invormativity index calculation'''
        H5 = variance + snr + len(des) - covariance
        print("H5", H5)
print()