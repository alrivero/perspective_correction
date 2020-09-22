import cv2
import json
import sys
import numpy as np
from getopt import getopt
from os import listdir


def define_board_points(board_dim):
    board_points = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
    board_points[:, :2] = (
        np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2))

    return board_points


def find_calibration_data(img_dir, board_dim):
    board_points = define_board_points(board_dim)  # 3D points searched
    obj_points = []  # 3D points in real world space of estimated board points
    img_points = []  # 2D points in image plane of estimated board points

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = listdir(img_dir)
    for file_name in images:
        img = cv2.imread(img_dir + "/" + file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_dim, None)

        # If found, add object points, image points (after refining them)
        if ret:
            obj_points.append(board_points)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

    # Return the points we searched for, 2D points found, and the images'
    # dimmensions.
    return (obj_points, img_points, gray.shape[::-1])


def calibrate_camera(calib_dataset):
    obj_points, img_points, img_dim = calib_dataset
    return cv2.calibrateCamera(obj_points, img_points, img_dim, None, None)


def find_mean_square_calibration_error(calib_dataset, calibration):
    obj_points, img_points, img_dim = calib_dataset
    _, mtx, dist, rvecs, tvecs = calibration

    total_error = 0
    for i in range(len(obj_points)):
        # Project the chessboard points we're looking for onto the camera plane
        # and determine by how much they differ from those found using
        # calibration.
        img_points_2, _ = cv2.projectPoints(
            obj_points[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )
        error = cv2.norm(img_points[i], img_points_2, cv2.NORM_L2)
        error /= len(img_points_2)

        total_error += error

    return total_error / len(obj_points)


def compute_camera_calibration(img_dir, board_dim):
    calib_dataset = find_calibration_data(img_dir, board_dim)
    calibration = calibrate_camera(calib_dataset)
    mean_error = find_mean_square_calibration_error(calib_dataset, calibration)

    return (calibration, mean_error)


if __name__ == "__main__":
    # Use getopt to gather our arguments
    img_dir = None
    save_dir = None
    board_dim_x = None
    board_dim_y = None

    opts, args = getopt(sys.argv[1:], "d:s:x:y:")
    for opt, arg in opts:
        if opt == "-d":
            img_dir = arg
        elif opt == "-s":
            save_dir = arg
        elif opt == "-x":
            board_dim_x = int(arg)
        elif opt == "-y":
            board_dim_y = int(arg)

    # Communicate the need for missing arguments
    if img_dir is None:
        print("Missing calibration image directory (-d)!")
        exit()
    if save_dir is None:
        print("Missing calibration save directory (-s)!")
        exit()
    if board_dim_x is None:
        print("Missing calibration board x dimmension (-x)!")
        exit()
    if board_dim_y is None:
        print("Missing calibration board y dimmension (-y)!")
        exit()

    # Process and save our camera calibration
    board_dim = (board_dim_x, board_dim_y)
    calibration, mean_error = compute_camera_calibration(img_dir, board_dim)
    _, k_mtx, dist, _, _ = calibration

    export_data = {
        "k_matrix": k_mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "mean_error": mean_error
    }
    with open(save_dir, "w") as saved_calibration:
        json.dump(export_data, saved_calibration)
