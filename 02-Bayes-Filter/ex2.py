#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Trung Kien
# Repo : https://github.com/ksvbka/robot-mapping

import numpy as np
import matplotlib.pyplot as plt


# Plot Belief
def plot_belief(belief):
    plt.figure()

    ax = plt.subplot(2, 1, 1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")

    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


# Motion Model
def motion_model(action, belief):
    p_correct = 0.7
    p_notmove = 0.2
    p_opposite = 0.1
    n = len(belief)
    u = 1 if action == "F" else -1

    belief_update = np.zeros(n)
    for i in range(n):
        # From current state, update belief of next state base on action u
        # NOTE:
        #   - at index 0 can't move backward
        #   - at index n-1 can't move forward
        idx_correct = i + u
        idx_notmove = i
        idx_opposite = i - u

        belief_update[idx_notmove] += p_notmove * belief[i]
        if idx_correct in range(n):
            belief_update[idx_correct] += p_correct * belief[i]
        if idx_opposite in range(n):
            belief_update[idx_opposite] += p_opposite * belief[i]

    # Normalization
    return belief_update/sum(belief_update)


# Observation/Sensor Model
def sensor_model(observation, belief, world):
    p_correct_white = 0.7
    p_correct_black = 0.9
    p_correct = p_correct_black if observation == 0 else p_correct_white
    p_wrong = p_correct_white if observation == 0 else p_correct_black

    assert len(belief) == len(world)
    belief_update = np.copy(belief)

    for i in range(len(world)):
        if world[i] == observation:
            belief_update[i] *= p_correct
        else:
            belief_update[i] *= (1 - p_wrong)

    # Normalization
    return belief_update/sum(belief_update)


# Recursive Bayes Filter
def recursive_bayes_filter(actions, observations, belief, world):
    # Initial position observation/sensor model
    belief_update = sensor_model(observations[0], belief, world)

    # Recursive calculation for each action
    for i, action in enumerate(actions):
        # Estimate using system model
        belief_estimate = motion_model(action, belief_update)
        # Correction base on sensor data
        belief_update = sensor_model(observations[i + 1], belief_estimate, world)

    return belief_update
