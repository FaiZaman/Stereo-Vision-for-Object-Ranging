# integration of stereo disparity and yolo

import cv2
import os
import sys
import argparse
import math
import numpy as np
import statistics
import yolo
import drawing as draw
import WLS_filter as WLS
from sklearn.preprocessing import normalize

master_path_to_dataset = "TTBB-durham-02-10-17-sub10/";
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

vehicles = ["person", "car", "bicycle", "truck", "motorbike", "aeroplane", "bus", "truck", "boat"]

keep_processing = True

# set to True to apply each - do not set both to True at the same time
WLS_on = True
sparse_ORB = False

# parse command line arguments for camera ID or video file
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')
args = parser.parse_args()

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
window_size = 3

def ORB(imgL, imgR, left, top, right, bottom):

    detected = False

    feature_object = cv2.ORB_create(800)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2

    (major, minor, _) = cv2.__version__.split(".")
    if ((int(major) >= 3) and (int(minor) >= 1)):
        search_params = dict(checks=50)   # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params,search_params)
    else:
        matcher = cv2.BFMatcher()

    # obtain yolo box as image 
    detected_boxL = imgL[top:bottom, left:right].copy()
    detected_boxR = imgR[top:bottom].copy()
    h, w, c = detected_boxL.shape

    # pad the boxes if they're too small
    if h < 80 or w < 80:
        detected_boxL = cv2.copyMakeBorder(detected_boxL, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        detected_boxR = cv2.copyMakeBorder(detected_boxR, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    if h > 0 and w > 0:
            
        detected = True
        
        # detect features and compute associated descriptor vectors
        left_keypoints, descriptors_cropped_region = feature_object.detectAndCompute(detected_boxL, None)

        # display keypoints on the image
        cropped_region_with_features = cv2.drawKeypoints(detected_boxL, left_keypoints, None, (255,0,0), 4)

    if detected:

        # detect and match features from current image
        right_keypoints, descriptors = feature_object.detectAndCompute(detected_boxR, None)

        matches = []

        if descriptors is None:
            descriptors = []

        if (len(descriptors) > 0):
                matches = matcher.knnMatch(descriptors_cropped_region, trainDescriptors = descriptors, k = 2)

        # Need to isolate only good matches, so create a mask
        # matchesMask = [[0,0] for i in range(len(matches))]
        # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
        # using the matching distances of the first and second matches

        good_matches = []
        disparity = []
        median_disparity = 0
        try:
            for (m, n) in matches:
                left_coord = left_keypoints[m.queryIdx].pt
                right_coord = right_keypoints[m.trainIdx].pt
                left_x = left_coord[1]
                left_y = left_coord[0]
                right_x = right_coord[1]
                right_y = right_coord[0]

                if left_x == right_x:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                        disparity.append(abs((left_y + left) - right_y))

        except ValueError:
            print("caught error - no matches from current frame")

        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), flags = 0)
        display_matches = cv2.drawMatches(detected_boxL, left_keypoints, detected_boxR, right_keypoints, good_matches, None, **draw_params)
        #cv2.imshow("Feature Matches", display_matches)

        if disparity == []:
            return 0
        median_disparity = statistics.median(disparity)

        return median_disparity


# get YOLO parameters
inpWidth, inpHeight, classes, net, output_layer_names = yolo.initialise(args.class_file, args.config_file, args.weights_file)

windowName = 'Stereo Vision for Object Ranging'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

# loop through all the images to play as video

for filename_left in left_file_list:

    # start a timer (to see how long processing and display takes)
    start_t = cv2.getTickCount()

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        #cv2.imshow('left image', imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        #cv2.imshow('right image', imgR)
        print();

        #equalised_grayL = cv2.equalizeHist(grayL)
        #equalised_grayR = cv2.equalizeHist(grayR)

        labL = cv2.cvtColor(imgL, cv2.COLOR_BGR2LAB)
        labR = cv2.cvtColor(imgR, cv2.COLOR_BGR2LAB)
        labL_planes = cv2.split(labL)
        labR_planes = cv2.split(labR)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        labL_planes[0] = clahe.apply(labL_planes[0])
        labR_planes[0] = clahe.apply(labR_planes[0])
        labL = cv2.merge(labL_planes)
        labR = cv2.merge(labR_planes)

        claheL = cv2.cvtColor(labL, cv2.COLOR_LAB2BGR)
        claheR = cv2.cvtColor(labR, cv2.COLOR_LAB2BGR)

        # crop the main car out using bitwise and to stop it from being detected in multiple images
        cropped_car_img = cv2.imread("cropped_car.png", cv2.IMREAD_COLOR)
        cropped_imgL = cv2.bitwise_and(claheL, cropped_car_img)
        cropped_imgR = cv2.bitwise_and(claheR, cropped_car_img)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(cropped_imgL, cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(cropped_imgR, cv2.COLOR_BGR2GRAY);

        if WLS_on:
            left_matcher, right_matcher = WLS.create_matcher(window_size)
            disparity_scaled = WLS.filter(left_matcher, right_matcher, claheL, claheR)
        else:
            grayL = np.power(grayL, 0.75).astype('uint8');
            grayR = np.power(grayR, 0.75).astype('uint8');

            # compute disparity image from undistorted and rectified stereo images
            # that we have loaded
            # (which for reasons best known to the OpenCV developers is returned scaled by 16)

            disparity = stereoProcessor.compute(grayL, grayR);
            
            # filter out noise and speckles (adjust parameters as needed)

            dispNoiseFilter = 5; # increase for more agressive filtering
            cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

            # scale the disparity to 8-bit for viewing
            # divide by 16 and convert to 8-bit image (then range of values should
            # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
            # so we fix this also using a initial threshold between 0 and max_disparity
            # as disparity=-1 means no disparity available

            _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO);
            disparity_scaled = (disparity / 16.).astype(np.uint8);

            # crop disparity to chop out left part where there are with no disparity
            # as this area is not seen by both cameras and also
            # chop out the bottom area (where we see the front of car bonnet)

            if (crop_disparity):
                width = np.size(disparity_scaled, 1);
                disparity_scaled = disparity_scaled[0:390,135:width];

        #cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));

        classIDs, boxes = yolo.create_and_remove(cropped_imgL, claheL, inpWidth, inpHeight, net, output_layer_names)

        distances = []
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            if classes[classIDs[detected_object]] in vehicles:
                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                
                if sparse_ORB:  # run sparse
                    median_disparity = ORB(claheL, claheR, left, top, left + width, top + height)
                    if median_disparity != None:
                        distance = draw.drawSparsePred(claheL, classes[classIDs[detected_object]], left, top, left + width, top + height, (255, 178, 50), median_disparity)
                else:   # run dense
                    distance = draw.drawPred(claheL, classes[classIDs[detected_object]], left, top, left + width, top + height, (255, 178, 50), disparity_scaled)
                
                if distance != -1:
                   distances.append(distance)

        # print nearest scene object
        print(filename_left)
        if distances != []:
            min_distance = min(distances)
            print(filename_right + " : nearest detected scene object (" + str(min_distance) + "m)")
        else:
            print(filename_right + " : nearest detected scene object (" + "0m)")

        # display image
        cv2.imshow(windowName, claheL)

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # start the event loop + detect specific key strokes
        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        #key = cv2.waitKey()
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows

cv2.destroyAllWindows()
