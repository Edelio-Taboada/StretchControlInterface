import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import threading
import math
from collections import deque

# Initialize the HSV lower and upper bounds
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])
min_contour_area = 500  # Minimum contour area to consider
buffer_size = 10  # Size of the moving average buffer

# Initialize buffers for moving average of centroids
centroid_buffer = deque(maxlen=buffer_size)

def nothing(x):
    pass

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_EXPOSURE, 40)

stop_event = threading.Event()

robot_x = 150
robot_y = 150
robot_angle = 0
velocity = 0
turn_rate = 0

def update_frame():
    while not stop_event.is_set():
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
        
        # Apply Gaussian blur to the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Apply morphological operations
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        # Filter contours by area and get up to two largest ones
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        
        for contour in valid_contours:
            # Calculate the centroid of each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        
        # Pad centroids to ensure a fixed length list
        while len(centroids) < 2:
            centroids.append((0, 0))

        # Smooth the centroids using moving average
        if centroids:
            centroid_buffer.append(centroids)
            valid_centroids = [c for c in centroid_buffer if c != [(0, 0), (0, 0)]]
            if valid_centroids:
                avg_centroids = np.mean(valid_centroids, axis=0).astype(int)
                centroids = [(int(c[0]), int(c[1])) for c in avg_centroids]
        
        # Update the GUI with the detected coordinates
        if centroids:
            root.after(0, update_gui, centroids)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk

# Function to update the GUI with the detected coordinates
def update_gui(centroids):
    global velocity, turn_rate

    canvas.delete("circles")
    canvas.delete("vector")
    info_canvas.delete("all")
    sim_canvas.delete("all")

    frame_height = 480
    middle_y = frame_height // 2

    for (cX, cY) in centroids:
        if cX != 0 and cY != 0:
            canvas.create_oval(cX-10, cY-10, cX+10, cY+10, fill="red", tags="circles")
    
    # If two objects are detected, draw a vector and calculate the angle
    if len(centroids) == 2 and all(cX != 0 and cY != 0 for cX, cY in centroids):
        # Sort centroids by y-coordinate
        centroids.sort(key=lambda x: x[1])
        (cX1, cY1), (cX2, cY2) = centroids
        canvas.create_line(cX1, cY1, cX2, cY2, fill="blue", width=2, tags="vector")

        # Calculate the angle with the vertical axis
        dx = cX2 - cX1
        dy = cY2 - cY1
        angle = math.degrees(math.atan2(dy, dx))
        vertical_angle = 90 - angle if angle >= 0 else -(90 + angle)
        print(f"Angle with the vertical axis: {vertical_angle:.2f} degrees")
        
        if abs(vertical_angle) > 30:
            info_canvas.create_text(150, 30, text="Angle > 30 degrees off vertical: YES", fill="red", font=("Helvetica", 14))
            turn_rate = vertical_angle / 90  # Turn rate proportional to the angle
        else:
            info_canvas.create_text(150, 30, text="Angle > 30 degrees off vertical: NO", fill="green", font=("Helvetica", 14))
            turn_rate = 0  # Go straight
    else:
        info_canvas.create_text(150, 30, text="Angle data unavailable", fill="grey", font=("Helvetica", 14))
        turn_rate = 0

    # Update the vertical position of the top object
    if centroids and centroids[0][0] != 0:
        top_cY = centroids[0][1]
        vertical_position = top_cY - middle_y
        info_canvas.create_text(150, 70, text=f"Vertical Position: {vertical_position}", fill="blue", font=("Helvetica", 14))
        
        # Draw the vertical position indicator
        box_middle = 75  # Center of the box
        box_height = 100  # Total height of the box
        box_width = 40    # Width of the box
        top_line = box_middle - box_height // 2
        bottom_line = box_middle + box_height // 2
        
        info_canvas.create_rectangle(130, top_line, 170, bottom_line, outline="black")
        info_canvas.create_line(130, box_middle, 170, box_middle, fill="black")

        if vertical_position > 0:
            info_canvas.create_rectangle(130, box_middle, 170, box_middle + vertical_position, fill="blue")
        else:
            info_canvas.create_rectangle(130, box_middle + vertical_position, 170, box_middle, fill="blue")

        # Set velocity based on vertical position
        velocity = vertical_position / 100  # Scale velocity down
    else:
        velocity = 0

    # Update the robot's position in the simulation
    update_simulation()

def update_simulation():
    global robot_x, robot_y, robot_angle, velocity, turn_rate

    # Update robot's position
    robot_x += velocity * math.cos(math.radians(robot_angle))
    robot_y += velocity * math.sin(math.radians(robot_angle))
    robot_angle += turn_rate

    # Constrain robot's position within the simulation window
    robot_x = max(20, min(robot_x, 280))
    robot_y = max(20, min(robot_y, 280))

    # Draw the robot
    robot_size = 20
    wheel_size = 5

    sim_canvas.create_rectangle(robot_x - robot_size, robot_y - robot_size, robot_x + robot_size, robot_y + robot_size, fill="gray")
    
    # Calculate wheel positions based on robot angle
    left_wheel_x = robot_x + robot_size * math.cos(math.radians(robot_angle + 90))
    left_wheel_y = robot_y + robot_size * math.sin(math.radians(robot_angle + 90))
    right_wheel_x = robot_x + robot_size * math.cos(math.radians(robot_angle - 90))
    right_wheel_y = robot_y + robot_size * math.sin(math.radians(robot_angle - 90))

    sim_canvas.create_rectangle(left_wheel_x - wheel_size, left_wheel_y - wheel_size, left_wheel_x + wheel_size, left_wheel_y + wheel_size, fill="black")
    sim_canvas.create_rectangle(right_wheel_x - wheel_size, right_wheel_y - wheel_size, right_wheel_x + wheel_size, right_wheel_y + wheel_size, fill="black")

    # Redraw the simulation canvas
    root.after(100, update_simulation)

# Create a Tkinter window for the GUI
root = tk.Tk()
root.title("Object Detection GUI")
canvas = Canvas(root, width=640, height=480)
canvas.pack()

# Create a separate Tkinter window for displaying the vertical position and angle status
info_window = tk.Toplevel(root)
info_window.title("Object Information")
info_canvas = Canvas(info_window, width=300, height=150)
info_canvas.pack()

# Create a separate Tkinter window for simulating the robot control
sim_window = tk.Toplevel(root)
sim_window.title("Robot Simulation")
sim_canvas = Canvas(sim_window, width=300, height=300)
sim_canvas.pack()

# Create trackbars for adjusting HSV ranges in a separate window
cv2.namedWindow("HSV Adjustments")
cv2.resizeWindow("HSV Adjustments", 600, 300)
cv2.createTrackbar("Lower H", "HSV Adjustments", 50, 179, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments", 70, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments", 100, 255, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments", 255, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments", 100, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments", 255, 255, nothing)

def on_closing():
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run OpenCV in a separate thread
threading.Thread(target=update_frame, daemon=True).start()

# Start the Tkinter main loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
