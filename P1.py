import matplotlib.pyplot as plt


import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import os
import sys


def process_frame(im):
    # convert to grayscale
    gs = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # denoise using gaussian
    cv.GaussianBlur(gs, (7, 7), 0)
    # run canny filter
    gs = cv.Canny(gs, 50, 150)
    # ROI
    mask = np.zeros_like(gs, dtype=np.uint8)
    sy, sx = mask.shape
    vertices = np.array([[0, sy - 1], [4 * sx / 10, sy / 2], [6 * sx / 10, sy / 2], [sx - 1, sy - 1]], np.int32)
    cv.fillPoly(mask, [vertices], 0xff)
    gs = cv.bitwise_and(gs, mask)

    # hough lines
    lines = cv.HoughLinesP(gs, 1, np.pi / 180, 80, 10)
    for x1, y1, x2, y2 in lines[0]:
        cv.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(lines)
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

image_file = file_list[0]
cv.startWindowThread()
im = cv.imread(os.path.join('test_images', image_file), cv.IMREAD_COLOR)

result = process_frame(im)
cv.imshow('input', im)
cv.imshow('result', result)

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

