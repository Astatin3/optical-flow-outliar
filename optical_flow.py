import numpy as np
import cv2

def detect_feature_points(frame, max_corners=1000):
    """
    Detect good features to track in the frame.
    """
    # Convert frame to grayscale if it's not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.001,
        minDistance=10,
        blockSize=7
    )
    
    return corners

def calculate_optical_flow(prev_frame, curr_frame, prev_points):
    """
    Calculate optical flow for given points between two frames.
    """
    # Convert frames to grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame
    
    # Calculate optical flow using Lucas-Kanade method
    curr_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Filter out points where flow wasn't found
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]
    
    return good_new, good_old

def estimate_camera_motion(prev_points, curr_points, threshold=5.0):
    """
    Estimate camera motion and identify outlier points.
    Returns mask of points that don't follow the dominant motion pattern.
    """
    # Calculate motion vectors
    motion_vectors = curr_points - prev_points
    
    # Calculate median motion as an estimate of camera motion
    median_motion = np.median(motion_vectors, axis=0)
    
    # Calculate the difference from median motion for each point
    motion_differences = np.linalg.norm(motion_vectors - median_motion, axis=1)
    
    # Calculate the median absolute deviation (MAD)
    mad = np.median(np.abs(motion_differences - np.median(motion_differences)))
    
    # Points with motion significantly different from the camera motion
    # are considered outliers (using modified z-score)
    outliers_mask = motion_differences > (threshold * mad)
    
    return outliers_mask

def analyze_motion(video_path):
    """
    Analyze motion in video and detect objects moving differently from camera motion.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read video")

    prev_points = None
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        
        # Detect initial points
        if prev_points is None:        
            prev_points = detect_feature_points(prev_frame)
        elif  curr_points.shape[0] < 600:
            prev_points = detect_feature_points(prev_frame)
        

        print(prev_points.shape)
                
        # Calculate optical flow
        curr_points, prev_points_matched = calculate_optical_flow(
            prev_frame, curr_frame, prev_points
        )
        
        if len(curr_points) > 0 and len(prev_points_matched) > 0:
            # Find points not moving with camera
            outliers_mask = estimate_camera_motion(prev_points_matched, curr_points)
            
            # Visualize results
            frame_vis = curr_frame.copy()
            
            # Draw all tracked points
            for i, (new, old) in enumerate(zip(curr_points, prev_points_matched)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Draw line between old and new position
                color = (0, 0, 255) if outliers_mask[i] else (0, 255, 0)
                cv2.line(frame_vis, (int(c), int(d)), (int(a), int(b)), color, 2)
                cv2.circle(frame_vis, (int(a), int(b)), 3, color, -1)
            
            cv2.imshow('Frame', frame_vis)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        # Update for next iteration
        prev_frame = curr_frame.copy()
        prev_points = curr_points.reshape(-1, 1, 2)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_path = 0
    analyze_motion(video_path)