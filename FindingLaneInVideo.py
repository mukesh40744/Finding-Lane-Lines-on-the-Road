'''
Created on Jan 12, 2019

@author: Mukesh Kumar Singh
'''
import numpy as np
import cv2
import os, glob
import matplotlib.pyplot as plt


def showImages(images, cmap=None):
    cols = 2
    rows = (len(images) + 1) // cols
    
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return np.array([x1, y1, x2, y2])


def avgSlope(image, lines):
    leftLine = []
    left_weights = []  # (length,)
    rightLine = []
    right_weights = []  # (length,)
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if (x2 - x1)!=0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                leftLine.append((slope, intercept))
                left_weights.append((length))
            else:
                rightLine.append((slope, intercept))
                right_weights.append((length))
                
    left_lane = np.dot(left_weights, leftLine) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, rightLine) / np.sum(right_weights) if len(right_weights) > 0 else None        
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return np.array([left_line, right_line])


def cannyAlgo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def displayLines(image, lines):
    lineImage1 = np.zeros_like(image)
    if lines is None:
        print(lines)
    else: 
        try:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(lineImage1, (x1, y1), (x2, y2), (255, 0, 255), 10)
        except AttributeError:
            print("shape not found")
    return lineImage1


def regionSelectInLane(image):
    imshape = image.shape
    #poly = np.array([[(20, imshape[0]), (450, 310),(490, 310), (imshape[1], imshape[0])]], dtype=np.int32)
    poly = np.array([[(100, imshape[0]), (450, 310),(490, 310), (imshape[1], imshape[0])]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


def drawLaneOnImage(image):
    laneImage = np.copy(image)
    canyImage = cannyAlgo(laneImage)
    cropedImage = regionSelectInLane(canyImage)
    lines = cv2.HoughLinesP(cropedImage, 2, np.pi / 180, 15, np.array([]), minLineLength=5, maxLineGap=2)
    print(lines)
    try:
        avgLines = avgSlope(laneImage, lines)
        lineImage = displayLines(laneImage, avgLines)
        comboImage = cv2.addWeighted(image, 0.8, lineImage, 1, 0)
    except AttributeError:
            print("shape not found")   
    return comboImage

lane_images = []
test_images = [cv2.imread(path) for path in glob.glob('test_images/*.jpg')]

for image in test_images:
    lane_images.append(drawLaneOnImage(image))
showImages(lane_images)    

#cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
# cap=cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
cap = cv2.VideoCapture("test_videos/challenge.mp4")
while(cap.isOpened()):
    _, color_frame = cap.read()
    cv2.imshow("result", drawLaneOnImage(color_frame))
    cv2.waitKey(50)
                             
