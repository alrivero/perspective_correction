import cv2
import glob
import numpy as np

def define_board_points(board_dim):
  board_points = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
  board_points[:, :2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2)

  return board_points

def find_calibration_data(img_dir, board_dim):
  board_points = define_board_points(board_dim) # The 3d points we're looking for
  obj_points = [] # 3D points in real world space of estimated board points
  img_points = [] # 2D points in image plane of estimated board points

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  images = glob.glob(img_dir)
  for file_name in images:
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, board_dim, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
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
    img_points_2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points_2, cv2.NORM_L2) / len(img_points_2)

    total_error += error
  
  return total_error / len(obj_points)