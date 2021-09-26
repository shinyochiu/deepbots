import numpy as np
from deepbots.robots.controllers.robot_emitter_receiver_csv import \
    RobotEmitterReceiverCSV


def normalize_to_range(value, min, max, newMin, newMax):
    value = float(value)
    min = float(min)
    max = float(max)
    newMin = float(newMin)
    newMax = float(newMax)
    return (newMax - newMin) / (max - min) * (value - max) + newMax

class FindTargetRobot(RobotEmitterReceiverCSV):
    def __init__(self, n_rangefinders):
        super(FindTargetRobot, self).__init__()
        self.setup_rangefinders(n_rangefinders)
        self.setup_motors()

    def create_message(self):
        message = [self.robot.getName()]
        for rangefinder, orient in zip(self.rangefinders, self.so_orient):
            message.append(rangefinder.getValue())
            message.append(orient)

        return message

    def use_message_data(self, message):
        MAX_VEL = 1
        MAX_OMG = 1
        if "LEADER" in self.robot.getName():
            MAX_VEL = 1/4
            MAX_OMG = 1/3
        # Action 0 is foward
        foward = float(message[0])
        # Action 1 is left turn
        left = float(message[1])
        # Action 2 is right turn
        right = float(message[2])

        # map wheel speed to [0, 4.4]
        wheel_fl = 6.4 * foward
        wheel_fr = 6.4 * foward
        wheel_bl = 6.4 * foward
        wheel_br = 6.4 * foward

        self.motorSpeeds[0] = 1.75 + wheel_fl*MAX_VEL + 2 * (MAX_OMG * (right - left))
        self.motorSpeeds[1] = 1.75 + wheel_fr*MAX_VEL + 2 * (MAX_OMG * (left - right))
        self.motorSpeeds[2] = 1.75 + wheel_bl*MAX_VEL + 2 * (MAX_OMG * (right - left))
        self.motorSpeeds[3] = 1.75 + wheel_br*MAX_VEL + 2 * (MAX_OMG * (left - right))
        # Clip final motor speeds to [-6.28, 6.28] to be sure that motors get valid values
        self.motorSpeeds = np.clip(self.motorSpeeds, -5, 5)

        # Apply motor speeds
        self.frontLeftMotor.setVelocity(self.motorSpeeds[0])
        self.frontRightMotor.setVelocity(self.motorSpeeds[1])
        self.backLeftMotor.setVelocity(self.motorSpeeds[2])
        self.backRightMotor.setVelocity(self.motorSpeeds[3])

    def setup_rangefinders(self, n_rangefinders):
        # Sensor setup
        self.n_rangefinders = n_rangefinders
        self.rangefinders = []
        self.tofNames = ['tof_' + str(i) for i in range(self.n_rangefinders)]  # 'tof_0', 'tof_1',...,'tof_7', only consider front tof sensors
        self.so_orient = [-1.57081, -0.7853981633974483, -0.47123889803846897, -0.15707963267948966, 0.15707963267948966, 0.47123889803846897, 0.7853981633974483, 1.57075]
        for i in range(self.n_rangefinders):
            self.rangefinders.append(
                self.robot.getDevice(self.tofNames[i]))
            self.rangefinders[i].enable(self.timestep)

    def setup_motors(self):
        # Front Motor setup
        self.frontLeftMotor = self.robot.getDevice('front left wheel')
        self.frontRightMotor = self.robot.getDevice('front right wheel')
        self.frontLeftMotor.setPosition(float('inf'))
        self.frontRightMotor.setPosition(float('inf'))
        self.frontLeftMotor.setVelocity(0.0)
        self.frontRightMotor.setVelocity(0.0)

        # Back Motor setup
        self.backLeftMotor = self.robot.getDevice('back left wheel')
        self.backRightMotor = self.robot.getDevice('back right wheel')
        self.backLeftMotor.setPosition(float('inf'))
        self.backRightMotor.setPosition(float('inf'))
        self.backLeftMotor.setVelocity(0.0)
        self.backRightMotor.setVelocity(0.0)

        self.motorSpeeds = [0.0, 0.0, 0.0, 0.0]


robot_controller = FindTargetRobot(8)
robot_controller.run()
