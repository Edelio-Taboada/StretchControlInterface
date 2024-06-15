import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import threading
import math
from collections import deque

# Initialize the HSV lower and upper bounds for green and orange
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

min_contour_area = 500  # Minimum contour area to consider
buffer_size = 10  # Size of the moving average buffer

# Initialize buffers for moving average of centroids
green_centroid_buffer = deque(maxlen=buffer_size)
orange_centroid_buffer = deque(maxlen=buffer_size)

def nothing(x):
    pass

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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

        # Threshold the HSV image to get only green and orange colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Apply Gaussian blur and morphological operations in a single step
        green_mask = cv2.morphologyEx(cv2.GaussianBlur(green_mask, (5, 5), 0), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        orange_mask = cv2.morphologyEx(cv2.GaussianBlur(orange_mask, (5, 5), 0), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours in the mask
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_centroids = []
        orange_centroids = []

        # Filter green contours by area and get the largest one
        valid_green_contours = [c for c in green_contours if cv2.contourArea(c) > min_contour_area]
        if valid_green_contours:
            green_contour = max(valid_green_contours, key=cv2.contourArea)
            M = cv2.moments(green_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                green_centroids.append((cX, cY))
        
        # Filter orange contours by area and get the largest one
        valid_orange_contours = [c for c in orange_contours if cv2.contourArea(c) > min_contour_area]
        if valid_orange_contours:
            orange_contour = max(valid_orange_contours, key=cv2.contourArea)
            M = cv2.moments(orange_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                orange_centroids.append((cX, cY))
        
        # Smooth the centroids using moving average
        if green_centroids:
            green_centroid_buffer.append(green_centroids[0])
            avg_green_centroid = np.mean(green_centroid_buffer, axis=0).astype(int)
            green_centroids = [(int(avg_green_centroid[0]), int(avg_green_centroid[1]))]
        
        if orange_centroids:
            orange_centroid_buffer.append(orange_centroids[0])
            avg_orange_centroid = np.mean(orange_centroid_buffer, axis=0).astype(int)
            orange_centroids = [(int(avg_orange_centroid[0]), int(avg_orange_centroid[1]))]
        
        # Update the GUI with the detected coordinates
        if green_centroids and orange_centroids:
            root.after(0, update_gui, green_centroids[0], orange_centroids[0])

        # Draw the center line and dead zone
        center_line_color = (255, 0, 0)  # Blue
        dead_zone_color = (0, 255, 0)  # Green
        middle_y = frame.shape[0] // 2
        dead_zone_size = 10
        top_dead_zone = middle_y - dead_zone_size
        bottom_dead_zone = middle_y + dead_zone_size

        # Draw the center line
        cv2.line(frame, (0, middle_y), (frame.shape[1], middle_y), center_line_color, 1)

        # Draw the dead zone
        cv2.line(frame, (0, top_dead_zone), (frame.shape[1], top_dead_zone), dead_zone_color, 1)
        cv2.line(frame, (0, bottom_dead_zone), (frame.shape[1], bottom_dead_zone), dead_zone_color, 1)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk

# Function to update the GUI with the detected coordinates
def update_gui(green_centroid, orange_centroid):
    global velocity, turn_rate

    canvas.delete("circles")
    canvas.delete("vector")
    info_canvas.delete("all")
    sim_canvas.delete("all")

    frame_height = 240
    middle_y = frame_height // 2
    dead_zone = 10  # Dead zone around the middle

    if green_centroid:
        cX, cY = green_centroid
        if cX != 0 and cY != 0:
            canvas.create_oval(cX-5, cY-5, cX+5, cY+5, fill="green", tags="circles")
    
    if orange_centroid:
        cX, cY = orange_centroid
        if cX != 0 and cY != 0:
            canvas.create_oval(cX-5, cY-5, cX+5, cY+5, fill="orange", tags="circles")
    
    # Draw a vector and calculate the angle
    if green_centroid and orange_centroid:
        (cX1, cY1) = orange_centroid
        (cX2, cY2) = green_centroid
        canvas.create_line(cX1, cY1, cX2, cY2, fill="blue", width=2, tags="vector")

        # Calculate the angle with the vertical axis
        dx = cX2 - cX1
        dy = cY2 - cY1
        angle = math.degrees(math.atan2(dy, dx))
        vertical_angle = 90 - angle if angle >= 0 else -(90 + angle)
        print(f"Angle with the vertical axis: {vertical_angle:.2f} degrees")
        
        if abs(vertical_angle) > 30:
            info_canvas.create_text(150, 30, text="Angle > 30 degrees off vertical: YES", fill="red", font=("Helvetica", 12))
            # Scale turn_rate such that 30 degrees is no turn and 180 degrees is full turn
            turn_rate = (vertical_angle - 30) / 150
        else:
            info_canvas.create_text(150, 30, text="Angle > 30 degrees off vertical: NO", fill="green", font=("Helvetica", 12))
            turn_rate = 0  # Go straight
    else:
        info_canvas.create_text(150, 30, text="Angle data unavailable", fill="grey", font=("Helvetica", 12))
        turn_rate = 0

    # Update the vertical position of the top object
    if orange_centroid:
        top_cY = orange_centroid[1]
        vertical_position = top_cY - middle_y
        print(f"Vertical position (orange centroid): {top_cY} (relative to middle: {vertical_position})")
        info_canvas.create_text(150, 60, text=f"Vertical Position: {vertical_position}", fill="blue", font=("Helvetica", 12))
        
        # Draw the vertical position indicator
        box_middle = 75  # Center of the box
        box_height = 100  # Total height of the box
        box_width = 40    # Width of the box
        top_line = box_middle - box_height // 2
        bottom_line = box_middle + box_height // 2
        
        info_canvas.create_rectangle(130, top_line, 170, bottom_line, outline="black")
        info_canvas.create_line(130, box_middle, 170, box_middle, fill="black")

        if vertical_position > dead_zone:
            info_canvas.create_rectangle(130, box_middle, 170, box_middle + vertical_position, fill="blue")
        elif vertical_position < -dead_zone:
            info_canvas.create_rectangle(130, box_middle + vertical_position, 170, box_middle, fill="blue")
        else:
            vertical_position = 0  # Within dead zone

        # Set velocity based on vertical position
        velocity = vertical_position / 400  # Scale velocity down further
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
    robot_x = max(40, min(robot_x, 260))
    robot_y = max(40, min(robot_y, 260))

    # Draw the walls
    sim_canvas.create_rectangle(0, 0, 300, 300, outline="black", width=2)

    # Draw the robot
    robot_size = 20
    wheel_size = 5

    # Draw the robot's body
    sim_canvas.create_rectangle(robot_x - robot_size, robot_y - robot_size, robot_x + robot_size, robot_y + robot_size, fill="gray")

    # Draw the robot's front indicator
    front_x = robot_x + robot_size * math.cos(math.radians(robot_angle))
    front_y = robot_y + robot_size * math.sin(math.radians(robot_angle))
    sim_canvas.create_line(robot_x, robot_y, front_x, front_y, fill="red", width=2)

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
canvas = Canvas(root, width=320, height=240)
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
