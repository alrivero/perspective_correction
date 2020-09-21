import random
import numpy as np
from statistics import median


def find_lever_vector(line, k_inv):
    point_a = np.array([line[0][0], line[0][1], 1]).transpose()
    point_b = np.array([line[1][0], line[1][1], 1]).transpose()

    point_a = np.matmul(k_inv, point_a)
    point_b = np.matmul(k_inv, point_b)

    lever_vec = np.cross(point_a, point_b)
    return lever_vec / np.linalg.norm(lever_vec)


def least_median_of_squares(lines, calibration):
    _, mtx_k, _, _, _ = calibration
    k_inv = np.linalg.inv(mtx_k)

    least_residual_m = (float("inf"), None)
    for _ in range(100):
        line_1 = random.choice(lines)
        line_2 = random.choice(lines)
        line_1_index = lines.index(line_1)
        line_2_index = lines.index(line_2)

        lever_1 = find_lever_vector(line_1, k_inv)
        lever_2 = find_lever_vector(line_2, k_inv)

        plane_normal = np.cross(lever_1, lever_2)
        plane_normal /= np.linalg.norm(plane_normal)

        residuals = []
        for i in range(len(lines)):
            if i == line_1_index or i == line_2_index:
                continue

            lever_i = find_lever_vector(lines[i], k_inv)
            residual = np.abs(np.dot(lever_i, plane_normal))

            residuals.append(residual)

        residual_median = median(residuals)
        if residual_median < least_residual_m[0]:
            least_residual_m = (residual_median, plane_normal)

    return least_residual_m
