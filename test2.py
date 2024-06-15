import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe Hands with detailed configuration.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize robot simulation parameters
box_size = 500
robot_pos = [box_size // 2, box_size // 2]
robot_radius = 20
robot_width = 40
robot_length = 60
left_wheel_velocity = 0
right_wheel_velocity = 0
max_wheel_velocity = 10
robot_angle = 0
dead_zone_height = 50

# Create a blank image for the robot simulation
robot_simulation = np.zeros((box_size, box_size, 3), dtype=np.uint8)

def draw_velocity_bar(image, velocity, max_velocity, position, color):
    bar_height = 100
    bar_width = 20
    max_bar_height = bar_height // 2
    normalized_velocity = int((velocity / max_velocity) * max_bar_height)
    cv2.rectangle(image, (position[0], position[1] - max_bar_height), (position[0] + bar_width, position[1] + max_bar_height), (255, 255, 255), 2)
    if velocity >= 0:
        cv2.rectangle(image, (position[0], position[1]), (position[0] + bar_width, position[1] - normalized_velocity), color, -1)
    else:
        cv2.rectangle(image, (position[0], position[1]), (position[0] + bar_width, position[1] - normalized_velocity), color, -1)

# Initialize FPS calculation
prev_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image_height, image_width, _ = image.shape
    middle_y = image_height // 2
    dead_zone_top = middle_y - dead_zone_height // 2
    dead_zone_bottom = middle_y + dead_zone_height // 2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw horizontal line in the middle of the image
    cv2.line(image, (0, middle_y), (image_width, middle_y), (255, 255, 255), 2)
    # Draw dead zone
    cv2.rectangle(image, (0, dead_zone_top), (image_width, dead_zone_bottom), (255, 255, 255), 2)

    left_finger_y = None
    right_finger_y = None
    left_finger_x = None
    right_finger_x = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_label = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label
            pointer_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Draw all landmarks as white circles
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

            # Draw pointer finger tip as a larger red circle
            pointer_finger_x = int(pointer_finger_tip.x * image_width)
            pointer_finger_y = int(pointer_finger_tip.y * image_height)
            cv2.circle(image, (pointer_finger_x, pointer_finger_y), 10, (0, 0, 255), -1)
            cv2.putText(image, f'({pointer_finger_x}, {pointer_finger_y})', (pointer_finger_x, pointer_finger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw thumb tip as a larger white circle
            thumb_x = int(thumb_tip.x * image_width)
            thumb_y = int(thumb_tip.y * image_height)
            cv2.circle(image, (thumb_x, thumb_y), 5, (255, 255, 255), -1)

            # Get the y-coordinate of the pointer finger tip
            if hand_label == 'Left':
                left_finger_y = pointer_finger_y
                left_finger_x = pointer_finger_x
            elif hand_label == 'Right':
                right_finger_y = pointer_finger_y
                right_finger_x = pointer_finger_x

    # Update wheel velocities based on swapped hand positions and dead zone
    if left_finger_y is not None:
        if dead_zone_top < left_finger_y < dead_zone_bottom:
            right_wheel_velocity = 0
        else:
            right_wheel_velocity = max_wheel_velocity * (middle_y - left_finger_y) / middle_y
    else:
        right_wheel_velocity = 0

    if right_finger_y is not None:
        if dead_zone_top < right_finger_y < dead_zone_bottom:
            left_wheel_velocity = 0
        else:
            left_wheel_velocity = max_wheel_velocity * (middle_y - right_finger_y) / middle_y
    else:
        left_wheel_velocity = 0

    # Update robot position and orientation
    velocity = (left_wheel_velocity + right_wheel_velocity) / 2
    rotation = (right_wheel_velocity - left_wheel_velocity) / robot_width
    robot_angle += rotation
    robot_pos[0] += int(velocity * np.cos(robot_angle))
    robot_pos[1] += int(velocity * np.sin(robot_angle))

    # Ensure robot stays within the box
    robot_pos[0] = np.clip(robot_pos[0], robot_radius, box_size - robot_radius)
    robot_pos[1] = np.clip(robot_pos[1], robot_radius, box_size - robot_radius)

    # Draw the robot in the simulation window
    robot_simulation.fill(0)

    # Calculate robot corners
    front_dx = int(robot_length / 2 * np.cos(robot_angle))
    front_dy = int(robot_length / 2 * np.sin(robot_angle))
    side_dx = int(robot_width / 2 * np.sin(robot_angle))
    side_dy = int(robot_width / 2 * np.cos(robot_angle))

    corners = [
        (robot_pos[0] + front_dx - side_dx, robot_pos[1] + front_dy + side_dy),
        (robot_pos[0] + front_dx + side_dx, robot_pos[1] + front_dy - side_dy),
        (robot_pos[0] - front_dx + side_dx, robot_pos[1] - front_dy - side_dy),
        (robot_pos[0] - front_dx - side_dx, robot_pos[1] - front_dy + side_dy)
    ]

    # Draw the robot body
    cv2.polylines(robot_simulation, [np.array(corners, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw the wheels at the front of the robot
    front_left_wheel = (int(corners[0][0]), int(corners[0][1]))
    front_right_wheel = (int(corners[1][0]), int(corners[1][1]))
    cv2.circle(robot_simulation, front_left_wheel, 5, (0, 255, 0), -1)
    cv2.circle(robot_simulation, front_right_wheel, 5, (0, 255, 0), -1)

    # Label the wheels with 'L' and 'R'
    cv2.putText(robot_simulation, 'R', (front_left_wheel[0] - 10, front_left_wheel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(robot_simulation, 'L', (front_right_wheel[0] - 10, front_right_wheel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the front direction
    front_point = (int(robot_pos[0] + front_dx), int(robot_pos[1] + front_dy))
    cv2.line(robot_simulation, (robot_pos[0], robot_pos[1]), front_point, (0, 0, 255), 2)

    # Draw velocity bars
    draw_velocity_bar(robot_simulation, left_wheel_velocity, max_wheel_velocity, (box_size - 60, 50), (0, 255, 0))
    draw_velocity_bar(robot_simulation, right_wheel_velocity, max_wheel_velocity, (box_size - 30, 50), (0, 255, 0))

    # Draw line between pointer fingers and the angle
    if left_finger_x is not None and right_finger_x is not None:
        cv2.line(image, (left_finger_x, left_finger_y), (right_finger_x, right_finger_y), (0, 255, 0), 2)
        angle = math.degrees(math.atan2(right_finger_y - left_finger_y, right_finger_x - left_finger_x))
        angle_text = f'Angle: {angle:.2f} degrees'
        midpoint_x = (left_finger_x + right_finger_x) // 2
        midpoint_y = (left_finger_y + right_finger_y) // 2
        cv2.putText(image, angle_text, (midpoint_x, midpoint_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display FPS on the images
    cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(robot_simulation, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting images
    cv2.imshow('MediaPipe Hands', image)
    cv2.imshow('Robot Simulation', robot_simulation)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
