import matplotlib.pyplot as plt


import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import os
import sys


def process_frame(im):
    # convert to gray-scale
    gs = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # de-noise using gaussian
    cv.GaussianBlur(gs, (7, 7), 0)
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

    # Hough lines
    lines = cv.HoughLinesP(gs, 1, np.pi / 180, 80, 10, 20, 5)
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line[0]
        # extend line so it crosses min_line_y, max_line_y
        if y1 != y2:
            line_x_0 = int(((x1 - x2) * max_line_y + x2 * y1 - x1 * y2) / (y1 - y2))
            line_x_1 = int(((x1 - x2) * min_line_y + x2 * y1 - x1 * y2) / (y1 - y2))
            cv.line(im, (line_x_0, max_line_y), (line_x_1, min_line_y), (0, 255, 0), 2)
    print(len(lines))
    return gs

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

image_file = file_list[0]
cv.startWindowThread()
im = cv.imread(os.path.join('test_images', image_file), cv.IMREAD_COLOR)

result = process_frame(im)
cv.imshow('input', im)
cv.moveWindow('input', 300, 400)
cv.imshow('result', result)
cv.moveWindow('result', 300, 400)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

