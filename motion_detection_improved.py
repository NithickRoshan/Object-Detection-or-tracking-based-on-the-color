import cv2
import imutils
import numpy as np
import time

class MotionDetector:
    def __init__(self, min_area=500, blur_size=21, threshold_value=25, 
                 method='mog2', sensitivity=0.7):
        """
        Enhanced motion detection with multiple algorithms
        
        Args:
            min_area: Minimum contour area to detect as motion
            blur_size: Gaussian blur kernel size (must be odd)
            threshold_value: Binary threshold value
            method: 'mog2', 'knn', or 'frame_diff'
            sensitivity: Motion sensitivity (0.0-1.0), higher = more sensitive
        """
        self.min_area = min_area
        self.blur_size = blur_size
        self.threshold_value = threshold_value
        self.method = method
        self.sensitivity = sensitivity
        self.prev_frame = None
        
        # Initialize background subtraction algorithm
        if method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        elif method == 'knn':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=True
            )
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Resize for faster processing
        frame = imutils.resize(frame, width=800)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        
        return frame, blurred
    
    def detect_motion_mog2_knn(self, frame):
        """Motion detection using MOG2 or KNN background subtraction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Remove shadows (values 127 are shadows in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        return fg_mask
    
    def detect_motion_frame_diff(self, frame):
        """Motion detection using frame-to-frame difference"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        # Calculate difference between frames
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Adaptive threshold based on sensitivity
        threshold = int(self.threshold_value / self.sensitivity)
        _, motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        self.prev_frame = gray
        return motion_mask
    
    def filter_contours(self, contours, frame_shape):
        """Filter contours by multiple criteria for accuracy"""
        valid_contours = []
        height, width = frame_shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if area is too small
            if area < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if contour touches image edges (likely noise)
            if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Skip extreme aspect ratios (likely noise)
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Calculate solidity (contour area / bounding rect area)
            rect_area = w * h
            solidity = area / rect_area if rect_area != 0 else 0
            
            # Skip if solidity is too low (noise/fragmented)
            if solidity < 0.3:
                continue
            
            valid_contours.append(contour)
        
        return valid_contours
    
    def detect(self, frame):
        """Main detection method"""
        original_frame, blurred_frame = self.preprocess_frame(frame)
        
        # Get motion mask based on selected method
        if self.method in ['mog2', 'knn']:
            motion_mask = self.detect_motion_mog2_knn(blurred_frame)
        else:
            motion_mask = self.detect_motion_frame_diff(blurred_frame)
        
        # Find contours
        contours = cv2.findContours(
            motion_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        
        # Filter contours
        valid_contours = self.filter_contours(contours, original_frame.shape)
        
        # Draw results
        motion_detected = len(valid_contours) > 0
        text = "Motion Detected" if motion_detected else "No Motion"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)
        
        for contour in valid_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(
            original_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        return original_frame, motion_mask, motion_detected, len(valid_contours)


def main():
    """Main execution function"""
    print("Starting Motion Detection...")
    print("Press 'q' to quit | 's' to switch method | '+'/'-' to adjust sensitivity")
    
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Camera warm-up
    
    # Initialize detector with MOG2 (best accuracy)
    detector = MotionDetector(
        min_area=500,
        blur_size=21,
        threshold_value=25,
        method='mog2',  # 'mog2', 'knn', or 'frame_diff'
        sensitivity=0.7
    )
    
    methods = ['mog2', 'knn', 'frame_diff']
    current_method_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect motion
        result_frame, motion_mask, detected, count = detector.detect(frame)
        
        # Display additional info
        info_text = f"Method: {detector.method} | Objects: {count} | Sensitivity: {detector.sensitivity:.1f}"
        cv2.putText(
            result_frame,
            info_text,
            (10, result_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show frames
        cv2.imshow("Motion Detection", result_frame)
        cv2.imshow("Motion Mask", motion_mask)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Switch detection method
            current_method_idx = (current_method_idx + 1) % len(methods)
            new_method = methods[current_method_idx]
            detector.method = new_method
            detector.bg_subtractor = (
                cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
                if new_method == 'mog2'
                else cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
                if new_method == 'knn'
                else None
            )
            print(f"Switched to {new_method}")
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity
            detector.sensitivity = min(1.0, detector.sensitivity + 0.1)
            print(f"Sensitivity: {detector.sensitivity:.1f}")
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity
            detector.sensitivity = max(0.1, detector.sensitivity - 0.1)
            print(f"Sensitivity: {detector.sensitivity:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Motion Detection Stopped")


if __name__ == "__main__":
    main()
