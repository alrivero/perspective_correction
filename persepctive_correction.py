import numpy as np


def find_top_left_corner(img_dim, h_mtx):
    min_x = float("inf")
    min_y = float("inf")

    corner_1 = np.array([0, 0, 1]).transpose()
    corner_2 = np.array([img_dim[1], 0, 1]).transpose()
    corner_3 = np.array([0, img_dim[0], 1]).transpose()
    corner_4 = np.array([img_dim[1], img_dim[0], 1]).transpose()

    corners = [corner_1, corner_2, corner_3, corner_4]
    for corner in corners:
        h_corner = np.matmul(h_mtx, corner)

        min_x = min(h_corner[0]/h_corner[2], min_x)
        min_y = min(h_corner[1]/h_corner[2], min_y)

    return (int(min_x), int(min_y))


def find_bottom_right_corner(img_dim, h_mtx):
    max_x = float("-inf")
    max_y = float("-inf")

    corner_1 = np.array([0, 0, 1]).transpose()
    corner_2 = np.array([img_dim[1], 0, 1]).transpose()
    corner_3 = np.array([0, img_dim[0], 1]).transpose()
    corner_4 = np.array([img_dim[1], img_dim[0], 1]).transpose()

    corners = [corner_1, corner_2, corner_3, corner_4]
    for corner in corners:
        h_corner = np.matmul(h_mtx, corner)

        max_x = max(h_corner[0]/h_corner[2], max_x)
        max_y = max(h_corner[1]/h_corner[2], max_y)

    return (int(max_x), int(max_y))


def compute_homography(horizontal_m, vertical_m, diagonal_m, K, K_inv, img_dim):
    w_mtx = np.array([horizontal_m, vertical_m, diagonal_m]).transpose()
    w_mtx = np.linalg.inv(w_mtx)

    h_mtx = np.matmul(w_mtx, K_inv)
    h_mtx = np.matmul(K, h_mtx)

    top_left_corner = find_top_left_corner(img_dim, h_mtx)
    bottom_right_corner = find_bottom_right_corner(img_dim, h_mtx)

    h_mtx_t = np.identity(3)
    h_mtx_t[0, 2] = -top_left_corner[0]
    h_mtx_t[1, 2] = -top_left_corner[1]

    img_dim_new = (
        bottom_right_corner[0] - top_left_corner[0],
        bottom_right_corner[1] - top_left_corner[1]
    )

    return (np.matmul(h_mtx_t, h_mtx), img_dim_new)
