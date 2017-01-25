import matplotlib.pyplot as plt


import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import os
import sys


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img, (x1, y1), (x2, y2), color, thickness)


def filter_lines(lines, y_level_0, y_level_1, x_range_0, x_range_1):
    filtered_lines = list()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # find x coordinates, where line crosses y_level_0 and y_level_1
        if y1 != y2:
            x_level_0 = int(((x1 - x2) * y_level_0 + x2 * y1 - x1 * y2) / (y1 - y2))
            x_level_1 = int(((x1 - x2) * y_level_1 + x2 * y1 - x1 * y2) / (y1 - y2))
            # check if crossings are in range
            if x_range_0[0] < x_level_0 < x_range_0[1] and x_range_1[0] < x_level_1 < x_range_1[1]:
                filtered_lines.append([x_level_0, y_level_0, x_level_1, y_level_1])
    return np.array(filtered_lines)


def process_frame(im):
    # convert to gray-scale
    gs = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # de-noise using gaussian smoothing
    cv.GaussianBlur(gs, (5, 5), 0)

    # run canny filter
    gs = cv.Canny(gs, 50, 150)
    # ROI
    mask = np.zeros_like(gs, dtype=np.uint8)
    sy, sx = mask.shape
    max_line_y = int(sy - 1)
    min_line_y = int(3 * sy / 5)
    min_line_x = 0
    max_line_x = sx - 1
    vertices = np.array([[min_line_x, max_line_y],
                         [4 * sx / 10, min_line_y],
                         [6 * sx / 10, min_line_y],
                         [max_line_x, max_line_y]], np.int32)
    cv.fillPoly(mask, [vertices], 0xff)
    gs = cv.bitwise_and(gs, mask)

    # hough lines
    lines = cv.HoughLinesP(gs, 1, np.pi / 180, 40, 10, 10, 5)
    draw_lines(im, lines)
    #
    # left_lines = filter_lines(lines, max_line_y, min_line_y, [min_line_x, sx / 2 - 1], [4 * sx / 10, 6 * sx / 10])
    # right_lines = filter_lines(lines, max_line_y, min_line_y, [sx / 2 + 1, max_line_x], [4 * sx / 10, 6 * sx / 10])
    #
    # left_line = np.mean(left_lines, axis=0)
    # print('left', left_line)
    # right_line = np.mean(right_lines, axis=0)
    # print('right', right_line)
    # cv.line(im, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 2)
    # cv.line(im, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 2)

    return im

# read test images

# video_files = ['solidWhiteRight.mp4', 'solidYellowLeft.mp4', 'challenge.mp4']
#
# cap = cv.VideoCapture(video_files[2])
#
# cnt = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         cnt += 1
#         result = process_frame(frame)
#         cv.imshow('frame', result)
#         print(frame.shape)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#
#         break
# cap.release()
# cv.destroyAllWindows()
#
# print('number of frames %d' % cnt)q


file_list = os.listdir('test_images')

image_file = file_list[1]
cv.startWindowThread()
im = cv.imread(os.path.join('test_images', image_file), cv.IMREAD_COLOR)

result = process_frame(im)
cv.imshow('result', result)
cv.moveWindow('result', 300, 400)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

