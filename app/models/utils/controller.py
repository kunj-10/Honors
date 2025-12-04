import numpy as np

class SteeringController:
    """
    Converts predicted steering to robot wheel commands.
    """

    def __init__(self, max_angle=30):
        self.max_angle = np.radians(max_angle)

    def compute_wheel_speeds(self, angle):
        angle = np.clip(angle, -self.max_angle, self.max_angle)
        left = 1.0 - angle
        right = 1.0 + angle
        return left, right
