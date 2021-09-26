import math

import numpy as np


def normalize_to_range(value, min, max, newMin, newMax):
    value = float(value)
    min = float(min)
    max = float(max)
    newMin = float(newMin)
    newMax = float(newMax)
    return (newMax - newMin) / (max - min) * (value - min) + newMin


def get_distance_from_target(robot_node, target_node):
    robotCoordinates = robot_node.getField('translation').getSFVec3f()
    targetCoordinate = target_node.getField('translation').getSFVec3f()

    dx = robotCoordinates[0] - targetCoordinate[0]
    dz = robotCoordinates[2] - targetCoordinate[2]
    distanceFromTarget = math.sqrt(dx * dx + dz * dz)
    return distanceFromTarget


def get_angle_from_target(robot_node,
                          target_node,
                          is_true_angle=False,
                          is_abs=False):
    robotAngle = get_rotation(robot_node)

    robotCoordinates = robot_node.getField('translation').getSFVec3f()
    targetCoordinate = target_node.getField('translation').getSFVec3f()

    x_r = (targetCoordinate[0] - robotCoordinates[0])
    z_r = (targetCoordinate[2] - robotCoordinates[2])

    # robotWorldAngle = math.atan2(robotCoordinates[2], robotCoordinates[0])

    #robotAngle = wrap2pi(robotAngle)
    '''x_f = x_r * math.sin(robotAngle) - z_r * math.cos(robotAngle)

    z_f = x_r * math.cos(robotAngle) + z_r * math.sin(robotAngle)'''

    # print("x_f: {} , z_f: {}".format(x_f, z_f) )
    '''if is_true_angle:
        x_f = -x_f'''
    angleDif = wrap2pi(np.pi - math.atan2(x_r, z_r) + robotAngle)

    if is_abs:
        angleDif = abs(angleDif)

    return angleDif


def get_rotation(robot_node):
    rotation = robot_node.getField('rotation').getSFRotation()
    robotAngle = wrap2pi(rotation[1] * rotation[3])
    return robotAngle


def wrap2pi(ang):
    if abs(ang) > np.pi:
        ang -= np.sign(ang) * 2 * np.pi
    return ang
