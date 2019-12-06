import cv2
import math
import numpy as np

focal_length = 399.9745178222656 # in pixels
baseline = 0.2090607502 # in meters

# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, left, top, right, bottom, colour, disparity_img):

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