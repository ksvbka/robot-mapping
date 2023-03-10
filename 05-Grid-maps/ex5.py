#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Trung Kien
# Repo : https://github.com/ksvbka/robot-mapping

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

show_animation = False

def plot_gridmap(gridmap):
    gridmap = np.array(gridmap, dtype=np.float64)
    plt.figure()
    plt.imshow(gridmap, cmap='Greys', vmin=0, vmax=1)


def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap


def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros((pose.shape))
    new_pose[0:] = np.round(pose[0:]/map_res) + origin[0]
    new_pose[1:] = np.round(pose[1:]/map_res) + origin[1]
    return new_pose.astype(int)


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle +
                         (num_beams*angular_res), num_beams)[idx]
    points = np.array([ranges[idx]*np.cos(angles),
                       ranges[idx]*np.sin(angles)])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom


def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = w_P@r_points
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l


def prob2logodds(p):
    return np.log(p / (1 - p))


def logodds2prob(l):
    l = np.array(l, dtype=np.float128)
    prob = 1 - (1 / (1 + np.exp(l)))
    return prob


def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    line = bresenham(cell[0], cell[1], endpoint[0], endpoint[1])
    prob_values = [prob_free for _ in range(len(line) - 1)]

    prob_values.append(prob_occ)
    prob_values = np.array(prob_values).reshape((len(line), 1))
    inv_sensor_model = np.hstack((line, prob_values))
    return inv_sensor_model


def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    # Known Poses for the grid mapping
    poses = poses2cells(poses_raw, occ_gridmap, map_res)

    # Converting cell value to the log value
    occ_gridmap = prob2logodds(occ_gridmap)

    # Given Sensor range value for every pose
    for i in range(poses.shape[0]):
        ranges = ranges2cells(ranges_raw[i], poses_raw[i], occ_gridmap, map_res).transpose()

        # update the probability within the senor range.
        for j in range(ranges.shape[0]):
            inv_sensor_val = inv_sensor_model(poses[i], ranges[j], prob_occ, prob_free)

            # Update the cell
            for cell in inv_sensor_val:
                x, y, prob = cell
                # update the grid map by converting probiblity output
                # from the sensor to logvalue and add it to grid value.
                occ_gridmap[int(x), int(y)] += prob2logodds(prob) - prob2logodds(prior)

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            gridmap = logodds2prob(occ_gridmap)
            gridmap = np.array(gridmap, dtype=np.float64)
            plt.imshow(gridmap, cmap='Greys', vmin=0, vmax=1)
            plt.plot(poses[:i, 1], poses[:i, 0], '-b', alpha=0.5, label='trajectory', linewidth=1)
            plt.scatter(poses[i, 1], poses[i, 0], c='g', s=20, marker='o')
            plt.scatter(ranges[:, 1], ranges[:, 0], c='r', s=0.5)
            plt.pause(0.001)

    # The cell value are converted back probability.
    occ_gridmap = logodds2prob(occ_gridmap)

    return occ_gridmap

if __name__ == "__main__":
    show_animation = True

    map_size = 100
    map_res = 0.25

    prior = 0.50
    prob_occ = 0.90
    prob_free = 0.35

    ranges_raw = np.loadtxt("data/ranges.data", delimiter=',', dtype='float')
    poses_raw = np.loadtxt("data/poses.data", delimiter=',', dtype='float')

    # initialize gridmap
    occ_gridmap = init_gridmap(map_size, map_res)+prior

    plt.figure()

    gridmap = grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior)
    plot_gridmap(gridmap)
    plt.show()
