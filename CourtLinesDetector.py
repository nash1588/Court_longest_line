import cv2
from scipy.spatial import distance
import numpy as np
import os
import subprocess

from UnionFind import UnionFind

LSD_CMD_ARGS = ["lsd_1.6" + os.path.sep + "lsd", '-q', '5']


INFINITE_SLOPE = 1e6
PARALLEL_SLOPE_TOL = 0.09
SAME_LINE_PIXEL_DIST_TOL = 9
RATIO_OF_MIN_LENGTH = 0.05
EDGE_THRESHOLD = 10
EPS = 1e-6
MAX_DISTANCE_RATIO_BETWEEN_LINES = 0.2
MIN_LENGTH_FOR_LINE_RATIO = 0.1


RATIO_OF_MIN_LENGTH = 0.02
EDGE_THRESHOLD = 10

class CourtLinesDetector(object):
    def __init__(self,min_length_ratio=0.02,):

    def is_line_on_img_edge(x1, y1, x2, y2, w, h):
        return False



if __name__ == '__main__':
    vid_path = ".." + os.path.sep + "misc" + os.path.sep + "rishon_vs_maccabi_youth_1min" + ".mp4"
    os.path.pardir