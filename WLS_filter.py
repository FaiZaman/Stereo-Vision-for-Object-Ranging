import cv2
import numpy as np

def create_matcher(window_size):

    left_matcher = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = 160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize = 5,
        P1 = 8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2 = 32 * 3 * window_size ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = 63,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    return left_matcher, right_matcher


def filter(left_matcher, right_matcher, greyL, greyR):

    # WLS Filter params
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    
    # apply WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disparity_left = left_matcher.compute(greyL, greyR)  # .astype(np.float32)/16
    disparity_right = right_matcher.compute(greyR, greyL)  # .astype(np.float32)/16
    disparity_left = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    filteredDisparity = wls_filter.filter(disparity_left, greyL, None, disparity_right)

    filteredDisparity = cv2.normalize(src=filteredDisparity, dst=filteredDisparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredDisparity = np.uint8(filteredDisparity)
    return filteredDisparity
