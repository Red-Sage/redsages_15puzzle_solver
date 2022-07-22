import cv2
from time import time
import numpy as np


def get_brisk_key_points(im):
    # helper function to get keypoints

    brisk = cv2.BRISK_create() 
    kp, des = brisk.detectAndCompute(im, None)

    return kp, des


def get_orb_key_points(im):
    # helper function to get fast keypoints

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(im, None)

    return kp, des


def get_sift_key_points(im, mask=None):
    # helper 

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(im, mask)

    if des is not None:
        des = des.astype(np.uint8)

    return kp, des


def get_transform_matrix(kp1, des1, kp2, des2):
    # Takes in images, key points and descriptors and returns the transform
    # matrix.

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Make sure there is at lease one match
    if len(matches[0]) <= 1:
        return None

    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    min_match_count = 5
    if len(good) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        m, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        return m
    else:
        return None


def put_text(im, text, location):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    # shift base down and to the left
    text_size = cv2.getTextSize(text, font, fontScale, thickness)
    location[0] = int(location[0] - text_size[0][0]/2)
    location[1] = int(location[1] + text_size[0][1]/2)

    cv2.putText(im, text,
                location,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA
                )


def remove_key_point(kp, des, upper_left, lower_right, out=False):
    # Removes key points and corresponding descriptors that fall inside
    # (or outside if 'out==True') of the box defined up upper left and
    # lower right.
    

    idx_to_remove = []
    pt_to_remove = []
    
    for idx, pt in enumerate(kp):
        x = pt.pt[0]
        y = pt.pt[1]


        if np.logical_xor((x >= upper_left[0] and x <= lower_right[0]) and
                          (y >= upper_left[1] and y <= lower_right[1]),
                          out):
            
            pt_to_remove.append(pt)
            idx_to_remove.append(idx)
            
    des = np.delete(des, idx_to_remove, axis=0)

    for pt in pt_to_remove:
        kp.remove(pt)

    return kp, des

    

