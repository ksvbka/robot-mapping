#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Trung Kien
# Repo : https://github.com/ksvbka/robot-mapping

import math
import numpy as np
import matplotlib.pyplot as plt


def plot_map(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys')


# to convert cordinates in the gridmap to world cordinates
def map2world(grid, i, j, x_0):
    origin = [grid.shape[0]/2, grid.shape[1]/2]
    pose = [0, 0, 0]
    pose[0] = (i - origin[0])*0.01 + x_0[0]  # 0.01 resolution
    pose[1] = (j - origin[1])*0.01 + x_0[1]
    return pose


def inverse_motion_model(u):
    x0, y0, theta0 = u[0]
    xt, yt, thetat = u[1]

    rot1 = math.atan2(yt - y0, xt - x0) - theta0
    trans = math.hypot(xt - x0, yt - y0)
    rot2 = thetat - theta0 - rot1
    return rot1, trans, rot2


def prob_triangular(query, std):
    return max(0, (1/(math.sqrt(6)*std) - (abs(query)/(6 * (std**2)))))


def prob_normal(query, std):
    return (1/math.sqrt(2*math.pi*std**2))*math.exp(-0.5*query**2/std**2)


def motion_model_odometry(x0, xt, u_t, alpha):
    rot1, trans, rot2 = inverse_motion_model(u_t)
    rot1_hat, trans_hat, rot2_hat = inverse_motion_model([x0, xt])
    p1 = prob_triangular((rot1-rot1_hat), (alpha[0]*abs(rot1)+alpha[1]*trans))
    p2 = prob_triangular((trans-trans_hat),
              (alpha[2]*trans + alpha[3]*(abs(rot1)+abs(rot2))))
    p3 = prob_triangular((rot2 - rot2_hat), (alpha[0]*rot2 + alpha[1]*trans))
    return p1*p2*p3


def motion_model_odometry_normal(x0, xt, u_t, alpha):
    rot1, trans, rot2 = inverse_motion_model(u_t)
    rot1_hat, trans_hat, rot2_hat = inverse_motion_model([x0, xt])
    p1 = prob_normal((rot1-rot1_hat), (alpha[0]*abs(rot1)+alpha[1]*trans))
    p2 = prob_normal((trans-trans_hat),
              (alpha[2]*trans + alpha[3]*(abs(rot1)+abs(rot2))))
    p3 = prob_normal((rot2 - rot2_hat), (alpha[0]*rot2 + alpha[1]*trans))
    return p1*p2*p3


def sample_triangular(b):
    return (math.sqrt(6)/2)*(np.random.uniform(-b, b) + np.random.uniform(-b, b))


def sample_normal(b):
    return 0.5*sum(np.random.uniform(-b, b) for _ in range(12))


def sample_motion_model_odometry(x0, u_t, alpha):
    rot1, trans, rot2 = inverse_motion_model(u_t)
    rot1_hat = rot1 + sample_triangular(alpha[0]*abs(rot1) + alpha[1]*trans)
    trans_hat = trans + \
        sample_triangular(alpha[2]*trans + alpha[3] * (abs(rot1)+abs(rot2)))
    rot2_hat = rot2 + sample_triangular(alpha[0]*abs(rot2) + alpha[1]*trans)

    x = x0[0] + trans_hat*math.cos(x0[2]+rot1_hat)
    y = x0[1] + trans_hat*math.sin(x0[2]+rot1_hat)
    theta = x0[2] + rot1_hat + rot2_hat
    return [x, y, theta]


def sample_motion_model_odometry_normal(x0, u_t, alpha):
    rot1, trans, rot2 = inverse_motion_model(u_t)
    rot1_hat = rot1 + sample_normal(alpha[0]*abs(rot1) + alpha[1]*trans)
    trans_hat = trans + \
        sample_normal(alpha[2]*trans + alpha[3] * (abs(rot1)+abs(rot2)))
    rot2_hat = rot2 + sample_normal(alpha[0]*abs(rot2) + alpha[1]*trans)

    x = x0[0] + trans_hat*math.cos(x0[2]+rot1_hat)
    y = x0[1] + trans_hat*math.sin(x0[2]+rot1_hat)
    theta = x0[2] + rot1_hat + rot2_hat
    return [x, y, theta]
