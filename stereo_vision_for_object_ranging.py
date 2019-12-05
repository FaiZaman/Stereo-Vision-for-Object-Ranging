# integration of stereo disparity and yolo

import cv2
import os
import sys
import argparse
import math
import numpy as np
import statistics
from sklearn.preprocessing import normalize

master_path_to_dataset = "TTBB-durham-02-10-17-sub10/";
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

focal_length = 399.9745178222656 # in pixels
baseline = 0.2090607502 # in meters

vehicles = ["person", "car", "bicycle", "truck", "motorbike", "aeroplane", "bus", "truck", "boat"]

keep_processing = True

# parse command line arguments for camera ID or video file
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')
args = parser.parse_args()

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""#"1506942604.475373_L.png"; # set to timestamp to skip forward to

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

# WLS Filter params
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
# apply WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

def WLS_filter(greyL, greyR):

    disparity_left = left_matcher.compute(greyL, greyR)  # .astype(np.float32)/16
    disparity_right = right_matcher.compute(greyR, greyL)  # .astype(np.float32)/16
    disparity_left = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    filteredDisparity = wls_filter.filter(disparity_left, greyL, None, disparity_right)  # important to put "imgL" here!!!

    filteredDisparity = cv2.normalize(src=filteredDisparity, dst=filteredDisparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredDisparity = np.uint8(filteredDisparity)
    return filteredDisparity


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


def drawSparsePred(image, class_name, left, top, right, bottom, colour, disparity_value):

    # Draw a bounding box
    cv2.rectangle(image, (left, top), (right, bottom), colour, 2)

    # calculate the distance according to the stereo depth formula
    if disparity_value == 0:
        label = '%s' % (class_name)
        labelSize, line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(image, (left, top - labelSize[1]),
            (left + labelSize[0], top + line), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return -1
    distance = round(((focal_length * baseline)/disparity_value), 2)

    # construct label
    label = '%s: %.2f%s' % (class_name, distance, "m")

    # Display the label at the top of the bounding box
    labelSize, line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - labelSize[1]),
        (left + labelSize[0], top + line), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return distance


# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, disparity_img):

    # Draw a bounding box
    cv2.rectangle(image, (left, top), (right, bottom), colour, 2)
    centre_x = math.floor((left + right)/2)
    centre_y = math.floor((top + bottom)/2)

    # crop the disparity image and take the median 
    width = right - left
    height = bottom - top
    cropped_disparity = disparity_img[top + int(height/3):bottom - int(height/3), left + int(width/3):right - int(width/3)]
    median_disparity = np.median(cropped_disparity)

    #if classes[classIDs[detected_object]] != "person": take bottom half of vehicle
     #   centre_y = centre_y + math.floor(bottom/4)

    # calculate the distance according to the stereo depth formula
    disparity_value = disparity_img[centre_y][centre_x]
    if median_disparity == 0:
        label = '%s' % (class_name)
        labelSize, line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(image, (left, top - labelSize[1]),
            (left + labelSize[0], top + line), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return -1
    distance = round(((focal_length * baseline)/median_disparity), 2)

    # construct label
    label = '%s: %.2f%s' % (class_name, distance, "m")

    # Display the label at the top of the bounding box
    labelSize, line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - labelSize[1]),
        (left + labelSize[0], top + line), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return distance


# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# define display window name + trackbar

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

        # crop the main car out using bitwise and to stop it from being detected in multiple images
        cropped_car_img = cv2.imread("cropped_car.png", cv2.IMREAD_COLOR)
        cropped_imgL = cv2.bitwise_and(imgL, cropped_car_img)
        cropped_imgR = cv2.bitwise_and(imgR, cropped_car_img)
        
        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(cropped_imgL, cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(cropped_imgR, cv2.COLOR_BGR2GRAY);

        #disparity = WLS_filter(grayL, grayR)

        # equalise contrast using histogram equalisation  
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        #equalised_grayL = cv2.equalizeHist(grayL)
        #equalised_grayR = cv2.equalizeHist(grayR)

        clahe = cv2.createCLAHE()
        clahe_grayL = clahe.apply(grayL)
        clahe_grayR = clahe.apply(grayR)

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(clahe_grayL, clahe_grayR);
        
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

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        #cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(cropped_imgL, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(imgL, results, confThreshold, nmsThreshold)

        distances = []
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            if classes[classIDs[detected_object]] in vehicles:
                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                
                median_disparity = ORB(imgL, imgR, left, top, left + width, top + height)
                if median_disparity != None:
                    distance = drawSparsePred(imgL, classes[classIDs[detected_object]], left, top, left + width, top + height, (255, 178, 50), median_disparity)
                # collect distances for each scene object
                #distance = drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50), disparity_scaled)
                if distance != -1:
                   distances.append(distance)

        # print nearest scene object
        print(filename_left)
        if distances != []:
            min_distance = min(distances)
            print(filename_right + " : nearest detected scene object (" + str(min_distance) + "m)")
        else:
            print(filename_right + " : nearest detected scene object (" + "0m)")


        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image

        cv2.imshow(windowName, imgL)

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
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows

cv2.destroyAllWindows()
