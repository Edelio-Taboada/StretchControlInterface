import cv2
import numpy as np
import serial
import time

# Initialize serial communication with Arduino
# Update the COM port as necessary
# arduino = serial.Serial(port='/dev/cu.usbmodem1401', baudrate=115200, timeout=1)
# arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)

time.sleep(2)  # Wait for the connection to establish
# lower_green = np.array([lh, ls, lv])
# upper_green = np.array([uh, us, uv])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

def nothing(x):
    pass

# Function to send Y position to Arduino with custom protocol
def send_to_arduino(y_pos):
    message = f'p{y_pos}l'
    # arduino.write(message.encode())

# Create a window for the HSV adjustments
cv2.namedWindow("HSV Adjustments")
cv2.resizeWindow("HSV Adjustments", 600, 300)

# Create trackbars for adjusting HSV ranges
cv2.createTrackbar("Lower H", "HSV Adjustments", 50, 179, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments", 70, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments", 100, 255, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments", 255, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments", 100, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments", 255, 255, nothing)

# Initialize the camera
cap = cv2.VideoCapture(0)
# # Set the resolution to 640 width and 480 height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Set the resolution to 1920 width and 1080 height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
# Set exposure to a lower value to enhance reflective tape visibility
# cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # The value depends on your camera. You may need to experiment with this.
cap.set(cv2.CAP_PROP_EXPOSURE, 40) 

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current positions of the trackbars
        lh = cv2.getTrackbarPos("Lower H", "HSV Adjustments")
        uh = cv2.getTrackbarPos("Upper H", "HSV Adjustments")
        ls = cv2.getTrackbarPos("Lower S", "HSV Adjustments")
        us = cv2.getTrackbarPos("Upper S", "HSV Adjustments")
        lv = cv2.getTrackbarPos("Lower V", "HSV Adjustments")
        uv = cv2.getTrackbarPos("Upper V", "HSV Adjustments")

        # Define HSV range from trackbars
        lower_green = np.array([lh, ls, lv])
        upper_green = np.array([uh, us, uv])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Send the Y position to Arduino
                send_to_arduino(cY)
                print(f"Sent Y position: {cY}")

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)  # Optionally display the mask
        
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture and close serial
    cap.release()
    cv2.destroyAllWindows()
    # arduino.close()