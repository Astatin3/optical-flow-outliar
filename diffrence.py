import cv2
import numpy as np

def process_video(video_path, scale_factor=0.5, min_area=500):
    """
    Process video for motion detection.
    
    Args:
        video_path: Path to input video
        output_path: Path to save processed video
        scale_factor: Factor to downscale the frames
        min_area: Minimum contour area to be considered as motion
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Error reading first frame")
    
    # Process first frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_small = cv2.resize(prev_gray, None, fx=scale_factor, fy=scale_factor)
    
    while True:
        # Read current frame
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale
        curr_small = cv2.resize(curr_gray, None, fx=scale_factor, fy=scale_factor)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(curr_small, prev_small)
        
        # Apply threshold to difference
        _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill in holes
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Scale back up to original size
        motion_mask = cv2.resize(dilated, (frame_width, frame_height))
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw motion areas on original frame
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(curr_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Write frame to output video
        cv2.imshow("e",curr_frame)


        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update previous frame
        prev_small = curr_small
        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Example usage
    input_video = 0
    try:
        process_video(input_video)
        print("Motion detection completed successfully")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()