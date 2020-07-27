import subprocess

import cv2
from scipy.spatial import distance
import numpy as np
import os

from UnionFind import UnionFind

INFINITE_SLOPE = 1e6
PARALLEL_SLOPE_TOL = 0.09
SAME_LINE_PIXEL_DIST_TOL = 9
RATIO_OF_MIN_LENGTH = 0.05
EDGE_THRESHOLD = 10
EPS = 1e-6
MAX_DISTANCE_RATIO_BETWEEN_LINES = 0.2
MIN_LENGTH_FOR_LINE_RATIO = 0.1


LSD_CMD_ARGS = ["lsd_1.6" + os.path.sep + "lsd", '-q', '5']

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


def are_lines_adjacent(line1, line2, w, h):
    p1, p2 = (line1[0], line1[1]), (line1[2], line1[3])
    p3, p4 = (line2[0], line2[1]), (line2[2], line2[3])
    thres = MAX_DISTANCE_RATIO_BETWEEN_LINES * ((w + h) / 2)
    return distance.euclidean(p1, p3) < thres or distance.euclidean(p1, p4) < thres or \
           distance.euclidean(p2, p3) < thres or distance.euclidean(p2, p4) < thres


def is_line_in_bottom_part(y1, y2, img_h, bottom_percent=1/3):
    return (y1+y2)/2 > (1 - bottom_percent) * img_h


def filter_lines(mat, lengths, w, h):
    '''
    Filters out the redundant lines:
    - lines on the edges of the the picture within 5 pixels
    - lines with length less than x% of the picture size
    :param mat, lengths:
    :return: list of lines indexes to remove
    '''
    list = []
    for i in range(len(lengths)):
        d = lengths[i]
        x1, y1, x2, y2 = mat[i][0:4]
        if is_line_in_bottom_part(y1, y2, h, 0.3) or \
                d < min(w * RATIO_OF_MIN_LENGTH, h * RATIO_OF_MIN_LENGTH) or \
                is_line_on_img_edge(x1, y1, x2, y2, w, h):
            list.append(i)
    return list


def is_on_same_line(slope1, slope2, b1, b2):
    if abs(slope1 - slope2) > PARALLEL_SLOPE_TOL:
        return False
    dist = abs((b1 - b2) / np.sqrt(slope1 * slope2 + 1))
    return dist <= SAME_LINE_PIXEL_DIST_TOL


def get_image_info(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return height, width


def convert_line_to_2d_points(coords):
    return (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3]))


def get_all_lines_info(line_segment_info_path):
    arr = np.genfromtxt(line_segment_info_path, delimiter=' ')
    lines_mat = arr.astype(np.float)
    lines_info = np.zeros([len(lines_mat), 5])
    for i in range(len(lines_info)):
        p1, p2 = convert_line_to_2d_points(lines_mat[i])
        length = distance.euclidean(p1, p2)
        x1, y1, x2, y2 = lines_mat[i][0:4]
        y1 = -y1
        y2 = -y2
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        if abs(x2 - x1) < EPS:
            m = INFINITE_SLOPE
        else:
            m = (y2 - y1) / (x2 - x1)
        b = y_center - m * x_center
        lines_info[i] = x_center, y_center, m, b, length
    return lines_mat, lines_info


def draw_all_segments(image_path, lines_mat):
    n_lines = len(lines_mat)
    img = cv2.imread(image_path)
    for i in range(n_lines):
        a, b = convert_line_to_2d_points(lines_mat[i])
        cv2.line(img, a, b, (0, 0, 255), 2)
        cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def filter_redundant_lines(lines_mat, lines_info, width, height):
    lines_to_remove = filter_lines(lines_mat, lines_info[:, -1], width, height)
    lines_mat = np.delete(lines_mat, lines_to_remove, axis=0)
    lines_info = np.delete(lines_info, lines_to_remove, axis=0)
    # print("lines filtered: " + str(len(lines_to_remove)))
    # print("total lines after filtered: " + str(len(lines_mat)))

    return lines_mat, lines_info


def unite_all_segment_on_same_line(lines_mat, lines_info, width, height):
    n_lines = len(lines_info)
    uf = UnionFind(n_lines, lines_info[:, 0:2], lines_info[:, 2], lines_info[:, 3], lines_info[:, 4], lines_mat[:])

    for i in range(n_lines):
        for j in range(n_lines):
            if i == j or uf.find(i) == uf.find(j): continue
            l1 = lines_info[i]
            l2 = lines_info[j]
            l1_slope = uf.get_slope(i)
            l1_b = uf.get_b(i)
            l2_slope = uf.get_slope(j)
            l2_b = uf.get_b(j)
            if ((i == 100) and (j == 182)) or ((i == 100) and (j == 200)):
                pass
                # debug
            if is_on_same_line(l1_slope, l2_slope, l1_b, l2_b) and \
                    are_lines_adjacent(lines_mat[i][0:4], lines_mat[j][0:4], width, height):
                uf.union(i, j)
    return uf


def filter_unionfind_groups(uf, line_length_att_dict, width, height):
    # TODO: filter sets that are:
    # - horizontal lines in bottom part of screen
    # - edges distance is less than X
    # -
    return []


def detect_longest_horizontal_line(union_find_segments, image_path, width, height, draw=False):
    img = cv2.imread(image_path)
    uf = union_find_segments
    line_length_att = {}

    for my_set in uf.getDisjointSets().values():
        parent = uf.find(list(my_set)[0])
        line_length_att[parent] = (1 / 3) * uf.get_length(parent) + (2 / 3) * uf.get_edges_distance(parent)

    disqualified_list = []  # filter_unionfind_groups(segments_unionfind,line_length_att,width,height)
    longest_line_id = None
    for key, value in reversed(sorted(line_length_att.items(), key=lambda item: item[1])):
        parent_segment_of_line = key
        if parent_segment_of_line in disqualified_list:
            continue
        if longest_line_id is None:
            longest_line_id = key
        if draw is True:
            slope_in_python_img_coord = - uf.get_slope(parent_segment_of_line)
            x_center, y_center = uf.get_xy_centers(parent_segment_of_line)
            y_center_new_coord = - y_center
            b = int(y_center_new_coord - slope_in_python_img_coord * x_center)
            p1 = (0, b)
            p2 = (width, int(slope_in_python_img_coord * width + b))
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.imshow('image', img)
            cv2.waitKey()
    cv2.destroyWindow('image')
    x, y = uf.get_xy_centers(longest_line_id)
    return longest_line_id, x, -y, uf.get_slope(longest_line_id)


def draw_all_segments_groups(union_find_segments, image_path, lines_mat, width, height):
    img = cv2.imread(image_path)
    uf = union_find_segments
    line_length_att = {}
    for my_set in uf.getDisjointSets().values():
        parent = uf.find(list(my_set)[0])
        line_length_att[parent] = (1 / 3) * uf.get_length(parent) + (2 / 3) * uf.get_edges_distance(parent)

    prev_set = None
    for my_set in uf.getDisjointSets().values():
        current_parent = uf.find(list(my_set)[0])
        if line_length_att[current_parent] < MIN_LENGTH_FOR_LINE_RATIO * (width + height) / 2 and \
                uf.get_edges_distance(current_parent) < (width + height) / 10:
            pass
        if prev_set is not None:
            for k in prev_set:
                p1, p2 = convert_line_to_2d_points(lines_mat[k])
                cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.imshow('image', img)
        print(my_set)
        edges_coords = uf.get_edges_coords(current_parent)
        v1 = (int(edges_coords[0][0]), int(edges_coords[0][1]))
        v2 = (int(edges_coords[1][0]), int(edges_coords[1][1]))
        cv2.line(img, v1, v2, (0, 255, 0), 2)
        for i in my_set:
            p1, p2 = convert_line_to_2d_points(lines_mat[i])
            print("p1: (%d,%d), p2:(%d,%d)  slope: %f" % (p1[0], p1[1], p2[0], p2[1], lines_info[i][2]))
            cv2.line(img, p1, p2, (255, 0, 0), 2)
            cv2.imshow('image', img)
            cv2.waitKey()
        prev_set = my_set

    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_longest_horizontal_line_wrapper(image_path, results_path):
    height, width = get_image_info(image_path)
    lines_mat, lines_info = get_all_lines_info(results_path)
    lines_mat, lines_info = filter_redundant_lines(lines_mat, lines_info, width, height)
    segments_unionfind = unite_all_segment_on_same_line(lines_mat, lines_info, width, height)
    id, x_cen, y_cen, slope = detect_longest_horizontal_line(segments_unionfind, image_path, width, height, draw=False)
    return x_cen, y_cen, slope


def get_lines_results_txt(image_path):
    assert os.path.exists(image_path)
    path = os.path.dirname(os.path.abspath(image_path))
    image_filename = os.path.basename(os.path.splitext(image_path)[0])
    results_filename = image_filename + "_results.txt"
    if not os.path.exists(path + os.path.sep + results_filename):
        # create results file with lsd.exe
        img = cv2.imread(image_path)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path + os.path.sep + image_filename + ".PGM", gray_frame)
        args = LSD_CMD_ARGS + [path + os.path.sep + image_filename + ".PGM", path + os.path.sep + results_filename]
        subprocess.call(args)
        os.remove(path + os.path.sep + image_filename + ".PGM")
    return path + os.path.sep + results_filename


if __name__ == '__main__':
    image_id = "14"
    img_path = "lsd_1.6" + os.path.sep + "data" + os.path.sep + image_id + ".jpg"
    line_segment_info_path = get_lines_results_txt(img_path)
    height, width = get_image_info(img_path)
    lines_mat, lines_info = get_all_lines_info(line_segment_info_path)
    # draw_all_segments(img_path,lines_mat)
    lines_mat, lines_info = filter_redundant_lines(lines_mat, lines_info, width, height)
    segments_unionfind = unite_all_segment_on_same_line(lines_mat, lines_info, width, height)
    id, x_cen, y_cen, slope = detect_longest_horizontal_line(segments_unionfind, img_path, width, height, draw=True)
#    draw_all_segments_groups(segments_unionfind,img_path,lines_mat,width,height)
