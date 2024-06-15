import cv2
import mediapipe as mp
import numpy as np

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

# Create a blank image for the robot simulation
robot_simulation = np.zeros((box_size, box_size, 3), dtype=np.uint8)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    left_hand_y = None
    right_hand_y = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the hand label (Left/Right)
            hand_label = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

            # Get the y-coordinate of the wrist (landmark 0)
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height

            if hand_label == 'Left':
                left_hand_y = wrist_y
            elif hand_label == 'Right':
                right_hand_y = wrist_y

    # Update wheel velocities based on hand positions
    if left_hand_y is not None:
        left_wheel_velocity = max_wheel_velocity * (1 - left_hand_y / image_height)
    if right_hand_y is not None:
        right_wheel_velocity = max_wheel_velocity * (1 - right_hand_y / image_height)

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

    # Draw the front direction
    front_point = (int(robot_pos[0] + front_dx), int(robot_pos[1] + front_dy))
    cv2.line(robot_simulation, (robot_pos[0], robot_pos[1]), front_point, (0, 0, 255), 2)

    # Display the resulting images
    cv2.imshow('MediaPipe Hands', image)
    cv2.imshow('Robot Simulation', robot_simulation)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
