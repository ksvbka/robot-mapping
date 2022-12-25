#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Trung Kien
# Repo : https://github.com/ksvbka/robot-mapping

from math import sqrt, exp, pi
import matplotlib.pyplot as plt

def plot_map(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys')

def prob(query,std):
    return (1/sqrt(2 * pi * (std ** 2))) * exp(-0.5 * ((query ** 2)/(std ** 2)))

def landmark_observation_model(z, x, m, std):
    r = sqrt((x[0]-m[0])**2 + (x[1]-m[1])**2)
    p = prob(z-r, std)
    return p
