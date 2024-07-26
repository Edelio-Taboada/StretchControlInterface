#!/usr/bin/env python

import rospy
import cv2
import mediapipe as mp
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize CvBridge
bridge = CvBridge()

# Initialize ROS node
rospy.init_node('camera', anonymous=True)
finger_pub = rospy.Publisher('finger_positions', Float64MultiArray, queue_size=10)

def draw_position_bars(image, left_finger_y, right_finger_y, mid_height):
    bar_width = 20
    bar_height = 100
    bar_color = (0, 255, 0)

    # Define top-left corner of the bars
    bar_x = image.shape[1] - 50
    bar_y = 50

    # Draw the middle line
    cv2.line(image, (bar_x, bar_y), (bar_x + bar_width * 2 + 10, bar_y), (255, 255, 255), 2)

    # Calculate the position of the left finger bar
    if left_finger_y < mid_height:
        left_bar_height = int((mid_height - left_finger_y) / mid_height * bar_height)
        cv2.rectangle(image, (bar_x, bar_y - left_bar_height), (bar_x + bar_width, bar_y), bar_color, -1)
    else:
        left_bar_height = int((left_finger_y - mid_height) / mid_height * bar_height)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + left_bar_height), bar_color, -1)

    # Calculate the position of the right finger bar
    if right_finger_y < mid_height:
        right_bar_height = int((mid_height - right_finger_y) / mid_height * bar_height)
        cv2.rectangle(image, (bar_x + bar_width + 10, bar_y - right_bar_height), (bar_x + bar_width * 2 + 10, bar_y), bar_color, -1)
    else:
        right_bar_height = int((right_finger_y - mid_height) / mid_height * bar_height)
        cv2.rectangle(image, (bar_x + bar_width + 10, bar_y), (bar_x + bar_width * 2 + 10, bar_y + right_bar_height), bar_color, -1)

def publish_finger_positions(left_finger_y, right_finger_y, mid_height):
    left_pos = (left_finger_y - mid_height) / mid_height * 1.5
    right_pos = (right_finger_y - mid_height) / mid_height * 1.5
    msg = Float64MultiArray()
    msg.data = [left_pos, right_pos]
    finger_pub.publish(msg)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        rospy.logerr("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logerr("Error: Failed to capture image")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find hands
            results = hands.process(image)

            # Convert the image color back so it can be displayed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mid_height = image.shape[0] // 2
            left_finger_y = mid_height
            right_finger_y = mid_height

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine which hand it is (left or right)
                    handedness = hand_handedness.classification[0].label

                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the coordinates for the index finger (landmark 8)
                    index_finger = hand_landmarks.landmark[8]
                    h, w, c = image.shape
                    cx, cy = int(index_finger.x * w), int(index_finger.y * h)

                    # Display the coordinates for the index finger
                    cv2.putText(image, f'({cx}, {cy})', (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Update the position for the bars
                    if handedness == 'Left':
                        left_finger_y = cy
                    else:
                        right_finger_y = cy

            # Draw the position bars
            draw_position_bars(image, left_finger_y, right_finger_y, mid_height)

            # Publish the finger positions
            publish_finger_positions(left_finger_y, right_finger_y, mid_height)

            # Display the resulting frame
            cv2.imshow('Hand Landmarks', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC'
                break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
