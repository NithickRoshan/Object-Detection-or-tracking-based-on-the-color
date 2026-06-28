'''
import cv2
import imutils
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)
time.sleep(2)

# Background subtraction - better than single frame reference
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

min_area = 500
blur_size = (21, 21)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("Motion Detection Running... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for faster processing
    frame = imutils.resize(frame, width=800)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, blur_size, 0)
    
    # Background subtraction (adaptive - works much better than frame difference)
    fg_mask = bg_subtractor.apply(blurred)
    
    # Remove shadows and noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Morphological operations - reduce noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill gaps
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)  # Expand objects
    
    # Find contours
    contours = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    motion_detected = False
    object_count = 0
    
    # Filter and draw contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip small noise
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip contours at image edges (usually noise)
        if x == 0 or y == 0 or (x + w) == frame.shape[1] or (y + h) == frame.shape[0]:
            continue
        
        # Skip extreme aspect ratios
        aspect_ratio = float(w) / h if h != 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True
        object_count += 1
    
    # Display results
    text = f"Motion Detected - Objects: {object_count}" if motion_detected else "No Motion"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Motion Detection Stopped")
'''

import time

prev_time = 0

while True:

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
