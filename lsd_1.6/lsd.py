import cv2
from scipy.spatial import distance
import numpy as np
import os
from sympy import Point, Line

RATIO_OF_MIN_LENGTH = 0.02
EDGE_THRESHOLD = 10

def is_line_on_img_edge(x1, y1, x2, y2, w, h):
    if x1 < EDGE_THRESHOLD and x2 < EDGE_THRESHOLD:
        return True
    if x1 > (w - EDGE_THRESHOLD) and x2 > (w - EDGE_THRESHOLD):
        return True
    if y1 < EDGE_THRESHOLD and y2 < EDGE_THRESHOLD:
        return True
    elif y1 > (h - EDGE_THRESHOLD) and y2 > (h - EDGE_THRESHOLD):
        return True
    return False


def filter_lines(img, lines_mat, lines_length,w,h):
    '''
    Filters out the redundant lines:
    - lines on the edges of the the picture within 5 pixels
    - lines with length less than x% of the picture size
    :param lines_mat, lines_length:
    :return: list of lines indexes to remove
    '''
    list = []
    for i in range(len(lines_length)):
        d = lines_length[i][0]
        x1,y1,x2,y2 = lines_mat[i][0:4]
        if d < min(w*RATIO_OF_MIN_LENGTH, h*RATIO_OF_MIN_LENGTH):
            list.append(i)
        elif is_line_on_img_edge(x1,y1,x2,y2,w,h):
            list.append(i)
    return list


arr = np.genfromtxt("lsd_1.6" + os.path.sep + "heat_results.txt", delimiter=' ')
lines_mat = arr.astype(np.float)

lines_length = np.zeros([len(lines_mat), 2])
for i in range(len(lines_length)):
    a = (int(lines_mat[i][0]) , int(lines_mat[i][1]))
    b = (int(lines_mat[i][2]) , int(lines_mat[i][3]))
    length = distance.euclidean(a, b)
    lines_length[i][0] = length
    lines_length[i][1] = np.rad2deg(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180

img = cv2.imread("lsd_1.6" + os.path.sep + "heat_kings.jpg")
height, width, channels = img.shape

lines_to_remove = filter_lines(img,lines_mat,lines_length,width,height)
lines_mat = np.delete(lines_mat, lines_to_remove, axis=0)
lines_length = np.delete(lines_length, lines_to_remove, axis=0)
print(len(lines_to_remove))

indices = np.flip(lines_length[:,0].argsort(axis=0))
sorted_lines_lengths = lines_length[indices]
sorted_lines_mat = lines_mat[indices]

sorted_lines = np.concatenate((sorted_lines_mat,sorted_lines_lengths),axis=1)



for i in range(len(sorted_lines)):
    a = (int(sorted_lines[i][0]), int(sorted_lines[i][1]))
    b = (int(sorted_lines[i][2]), int(sorted_lines[i][3]))
    cv2.line(img, a, b, (0, 0, 255), 2)

hist = np.zeros(180)

for i in range(len(sorted_lines)):
    angle = round(sorted_lines[i][-1])
cv2.imshow('Detect line', img)
cv2.waitKey()
