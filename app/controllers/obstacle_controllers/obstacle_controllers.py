import sys
from pathlib import Path

# compute project root: .../Honors (adjust the number if your layout differs)
# file: .../app/controllers/obstacle_controllers/obstacle_controllers.py
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Hello1")
import math

from controller import Supervisor
from app.models.evaluate import predict

print("Hello")

# --- Constants ---
TIME_STEP = 32
MAX_SPEED = 6.4
CRUISING_SPEED = 5.0
TURN_MULTIPLIER = 3


# Obstacle Thresholds
# If obstacle score is higher than this, SWITCH to pure avoidance mode
AVOIDANCE_TRIGGER_THRESHOLD = 0.25
DECREASE_FACTOR = 0.9
BACK_SLOWDOWN = 0.9
RUN_ANGLE=False

# Goal Parameters
DISTANCE_TOLERANCE = 0.5
GOAL_GAIN = 2.0


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def main():
    # 1. Init Supervisor
    robot = Supervisor()

    # 2. Get Devices
    cmr = robot.getDevice("Sick LMS 291")
    cmr.enable(TIME_STEP)

    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)

    imu = robot.getDevice("inertial unit")
    imu.enable(TIME_STEP)

    fl = robot.getDevice("front left wheel")
    fr = robot.getDevice("front right wheel")
    bl = robot.getDevice("back left wheel")
    br = robot.getDevice("back right wheel")

    wheels = [fl, fr, bl, br]
    for w in wheels:
        w.setPosition(float("inf"))
        w.setVelocity(0.0)

    # 3. Get Goal
    goal_node = robot.getFromDef("GOAL")
    if goal_node is None:
        print("Error: DEF GOAL not found.")
        return
    goal_field = goal_node.getField("translation")

    # 4. Pre-compute Braitenberg Weights (From your code)
    lms_width = cmr.getHorizontalResolution()
    half_width = lms_width // 2
    max_range = cmr.getMaxRange()
    # Use a fixed threshold for indoor walls (e.g., 2.0 meters)
    # instead of max_range/20 which might be too short/long
    range_threshold = 2.0

    braitenberg = [gaussian(i, half_width, lms_width / 5) for i in range(lms_width)]

    if RUN_ANGLE:
        angle = [predict(camera.getImage())]

    print("Controller started: Priority Avoidance Mode")

    while robot.step(TIME_STEP) != -1:
        # --- SENSORS ---
        cmr_values = cmr.getRangeImage()
        current_pos = gps.getValues()
        rpy = imu.getRollPitchYaw()
        current_yaw = rpy[2]  # Adjust index if using ENU vs NED
        target_pos = goal_field.getSFVec3f()

        # --- OBSTACLE CALCULATION (Your Braitenberg Logic) ---
        left_obstacle = 0.0
        right_obstacle = 0.0

        for i in range(half_width):
            # Left side (check indices based on your cmr mounting)
            if cmr_values[i] < range_threshold:
                left_obstacle += braitenberg[i] * (1.0 - cmr_values[i] / max_range)

            # Right side
            j = lms_width - i - 1
            if cmr_values[j] < range_threshold:
                right_obstacle += braitenberg[i] * (1.0 - cmr_values[j] / max_range)

        obstacle_score = left_obstacle + right_obstacle

        # --- DECISION TREE ---

        left_speed = 0
        right_speed = 0

        # CASE 1: OBSTACLE DETECTED -> IGNORE GOAL, AVOID!
        if obstacle_score > AVOIDANCE_TRIGGER_THRESHOLD:
            # This logic comes directly from your provided snippet
            # It turns the robot AWAY from the obstacle
            speed_factor = (
                (1.0 - DECREASE_FACTOR * obstacle_score) * MAX_SPEED / obstacle_score
            )

            # If left obstacle is high, we want right wheel to speed up (turn left)?
            # Actually, standard Braitenberg "Fear":
            # High Sensor Left -> High Motor Left -> Turn Right

            # Applying your logic:
            left_speed = speed_factor * left_obstacle * TURN_MULTIPLIER
            right_speed = speed_factor * right_obstacle * TURN_MULTIPLIER

        # CASE 2: PATH CLEAR -> GO TO GOAL
        else:
            # Calculate Goal Vector
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]  # Using Y for Z-up world

            dist_to_goal = math.sqrt(dx * dx + dy * dy)

            if dist_to_goal < DISTANCE_TOLERANCE:
                print("GOAL REACHED!")
                left_speed = 0
                right_speed = 0
            else:
                target_angle = math.atan2(dy, dx)
                heading_error = normalize_angle(target_angle - current_yaw)

                # Simple P-Controller for heading
                turn = GOAL_GAIN * heading_error

                left_speed = CRUISING_SPEED - turn
                right_speed = CRUISING_SPEED + turn

        # --- ACTUATION ---

        # Clamp speeds
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        # Back wheels slightly slower (from your logic)
        bl_speed = BACK_SLOWDOWN * left_speed
        br_speed = BACK_SLOWDOWN * right_speed

        fl.setVelocity(left_speed)
        fr.setVelocity(right_speed)
        bl.setVelocity(bl_speed)
        br.setVelocity(br_speed)


if __name__ == "__main__":
    main()
