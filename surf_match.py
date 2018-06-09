import numpy
import cv2
from matplotlib import pyplot as plt

############### Image Matching ###############

#img1 = cv2.imread('image/template.jpg',0)
#img2 = cv2.imread('image/picture.jpg',0)

def match_images(img1, img2):
    """Given two images, returns the matches"""
    surf = cv2.xfeatures2d.SURF_create(500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = surf.detectAndCompute(img1, None)
    kp2, desc2 = surf.detectAndCompute(img2, None)

    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

############### Match Diplaying ###############

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    cv2.imshow(win, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)

    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])

    if len(kp_pairs) > 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, None
    if len(p1):
        explore_match(window_name, img1, img2, kp_pairs, status, H)
'''
############### Test ###############

img1 = cv2.imread('image/template.jpg', 0)
img2 = cv2.imread('image/picture.jpg', 0)
#cv2.imshow('REAL',img1)
#cv2.imshow('Rotated',img2)
#img_1= cv2.resize(img1,(500,600))
#img_2= cv2.resize(img2,(500,600))
kp_pairs = match_images(img1, img2)
matches = bf.match(des1,des2)

if kp_pairs:
    draw_matches('Matching Features', kp_pairs, img1, img2)
    plt.imshow(img1,img2), plt.show()
   # plt.imshow(img2), plt.show()
else:
    print ("No matches found")
