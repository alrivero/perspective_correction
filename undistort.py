import cv2

def undistort_img(distorted_img, calibration):
  h, w = distorted_img.shape[:2]
  _, mtx, dist, rvecs, tvecs = calibration

  # Undistort the distorted image
  new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
  undistorted_img = cv2.undistort(distorted_img, mtx, dist, None, new_camera_mtx)

  # Crop the undistorted image
  x, y, w, h = roi
  undistorted_img = undistorted_img[y:y+h, x:x+w]

  return undistorted_img
