
import math
import numpy as np
import cv2
from scipy.spatial import KDTree

BETA = 0.3
MIN_SIGMA = 4.0
MAX_SIGMA = 15.0
DEFAULT_SIGMA = 8.0

REFERENCE_RESOLUTION = (672, 512)

def generate_density_map(image_shape, points, base_sigma=DEFAULT_SIGMA,
                         reference_resolution=REFERENCE_RESOLUTION):
    h, w = image_shape
    density = np.zeros(image_shape, dtype=np.float32)
    num_points = len(points)

    if num_points == 0:
        return density

    points_array = np.array(points, dtype=np.float64)

    if num_points > 3:
        tree = KDTree(points_array)
        distances, _ = tree.query(points_array, k=4)

        avg_distances = np.mean(distances[:, 1:], axis=1)
        sigmas = BETA * avg_distances
        sigmas = np.clip(sigmas, MIN_SIGMA, MAX_SIGMA)
    else:
        sigmas = np.full(num_points, base_sigma, dtype=np.float32)

    for i, point in enumerate(points):
        pt_x, pt_y = int(point[0]), int(point[1])
        sigma = sigmas[i]

        k_size = int(3 * sigma)

        y_grid, x_grid = np.ogrid[-k_size:k_size + 1, -k_size:k_size + 1]
        H = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))

        H_sum = H.sum()
        if H_sum == 0:
            continue

        H = H / H_sum

        y1, y2 = pt_y - k_size, pt_y + k_size + 1
        x1, x2 = pt_x - k_size, pt_x + k_size + 1

        k_y1, k_y2 = 0, 2 * k_size + 1
        k_x1, k_x2 = 0, 2 * k_size + 1

        if y1 < 0:
            k_y1 = -y1
            y1 = 0
        if y2 > image_shape[0]:
            k_y2 -= (y2 - image_shape[0])
            y2 = image_shape[0]

        if x1 < 0:
            k_x1 = -x1
            x1 = 0
        if x2 > image_shape[1]:
            k_x2 -= (x2 - image_shape[1])
            x2 = image_shape[1]

        if y1 < y2 and x1 < x2:
            density[y1:y2, x1:x2] += H[k_y1:k_y2, k_x1:k_x2]

    return density

def create_visualization(image, density_map, colormap=cv2.COLORMAP_JET,
                         alpha=0.5, input_is_rgb=False):
    if density_map.max() > 0:
        density_norm = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(density_norm, colormap)

    if input_is_rgb:
        heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    else:
        heatmap = heatmap_bgr

    overlay = cv2.addWeighted(image, alpha, heatmap, 1.0 - alpha, 0)

    return overlay
