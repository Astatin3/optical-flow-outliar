import numpy as np
import fcntl
import mmap
import struct
import time
import os

class FrameBuffer:
    """Class to handle direct framebuffer access on Raspberry Pi."""
    
    def __init__(self, device='/dev/fb0'):
        # Open the framebuffer device
        self.fb = open(device, 'rb+')
        
        # Get fixed screen information
        fix_info = struct.unpack('IIIIHH',
            fcntl.ioctl(self.fb.fileno(),
                       0x4602,  # FBIOGET_FSCREENINFO
                       struct.pack('IIIIHH', 0, 0, 0, 0, 0, 0)))
        
        # Get variable screen information
        var_info = struct.unpack('IIIIIIIIHHHHHH',
            fcntl.ioctl(self.fb.fileno(),
                       0x4600,  # FBIOGET_VSCREENINFO
                       struct.pack('IIIIIIIIHHHHHH', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        
        self.xres = var_info[0]
        self.yres = var_info[1]
        self.bits_per_pixel = var_info[6]
        self.bytes_per_pixel = self.bits_per_pixel // 8

        print(f"Screen size: {self.xres}x{self.yres}")
        
        # Map the framebuffer to memory
        self.fb_size = fix_info[1]  # Length of framebuffer memory
        self.fb_map = mmap.mmap(self.fb.fileno(), self.fb_size, mmap.MAP_SHARED, mmap.PROT_WRITE|mmap.PROT_READ)
    
    def display_frame(self, frame):
        """Display a numpy array as a frame.
        
        Args:
            frame: numpy array of shape (height, width, 3) with uint8 RGB values
        """
        # Ensure frame matches framebuffer dimensions
        if frame.shape[:2] != (self.yres, self.xres):
            frame = cv2.resize(frame, (self.xres, self.yres))
        
        # Convert frame to correct format (BGR888 to RGB565)
        if self.bits_per_pixel == 16:
            # Convert RGB888 to RGB565
            r = (frame[:, :, 0] >> 3).astype(np.uint16) << 11
            g = (frame[:, :, 1] >> 2).astype(np.uint16) << 5
            b = (frame[:, :, 2] >> 3).astype(np.uint16)
            frame_rgb565 = (r | g | b).astype(np.uint16)
            buffer = frame_rgb565.tobytes()
        else:
            # Assume 24/32 bit color
            buffer = frame.tobytes()
        
        # Write to framebuffer
        self.fb_map.seek(0)
        self.fb_map.write(buffer)
    
    def __del__(self):
        self.fb_map.close()
        self.fb.close()




class PitchYawHUD:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Colors (in BGR for OpenCV)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.BLACK = (0, 0, 0)
        
        # HUD dimensions
        self.YAW_HEIGHT = 40
        self.PITCH_WIDTH = 40
        
        # Tick marks
        self.MAJOR_TICK_LENGTH = 15
        self.MINOR_TICK_LENGTH = 8
        self.LABEL_OFFSET = 20
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4
        
        # Cardinal directions for yaw
        self.cardinal_directions = {
            0: "N", 45: "NE", 90: "E", 135: "SE",
            180: "S", 225: "SW", 270: "W", 315: "NW"
        }
    
    def draw_text(self, img, text, pos, color=None):
        if color is None:
            color = self.WHITE
        cv2.putText(img, text, pos, self.font, self.font_scale, color, 1, cv2.LINE_AA)
    
    def draw_yaw_indicator(self, img, yaw_angle):
        # Normalize yaw angle to 0-360
        yaw_angle = yaw_angle % 360
        
        # Draw background
        # cv2.rectangle(img, (0, 0), (self.width, self.YAW_HEIGHT), self.GRAY, -1)
        
        # Calculate pixel per degree for yaw
        pixels_per_degree = self.width / 360
        
        # Draw tick marks
        for angle in range(0, 360, 5):  # Draw every 5 degrees
            x_pos = int((angle - yaw_angle) * pixels_per_degree)
            x_pos = x_pos % self.width
            
            # Determine tick length
            if angle % 45 == 0:  # Cardinal and intercardinal directions
                tick_length = self.MAJOR_TICK_LENGTH
                # Draw direction label
                if angle in self.cardinal_directions:
                    text = self.cardinal_directions[angle]
                    text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
                    text_x = int(x_pos - text_size[0]/2)
                    self.draw_text(img, text, (text_x, self.LABEL_OFFSET + text_size[1]))
            else:
                tick_length = self.MINOR_TICK_LENGTH
            
            # Draw tick
            cv2.line(img, (x_pos, 0), (x_pos, tick_length), self.WHITE, 1, cv2.LINE_AA)
    
    def draw_pitch_indicator(self, img, pitch_angle):
        # Normalize pitch angle to -180 to 180
        pitch_angle = max(-180, min(180, pitch_angle))
        
        # Draw background
        # cv2.rectangle(img, 
        #              (self.width - self.PITCH_WIDTH, 0),
        #              (self.width, self.height),
        #              self.GRAY, -1)
        
        # Calculate pixel per degree for pitch
        pixels_per_degree = self.height / 360
        
        # Draw tick marks
        for angle in range(-180, 181, 10):  # Draw every 10 degrees
            y_pos = int(self.height/2 + (angle - pitch_angle) * pixels_per_degree)
            
            # Skip if outside screen
            if y_pos < 0 or y_pos > self.height:
                continue
            
            # Determine tick length
            if angle % 30 == 0:  # Major ticks
                tick_length = self.MAJOR_TICK_LENGTH
                # Draw angle label
                text = str(angle)
                text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
                text_x = self.width - self.PITCH_WIDTH - text_size[0] - 5
                text_y = int(y_pos + text_size[1]/2)
                self.draw_text(img, text, (text_x, text_y))
            else:
                tick_length = self.MINOR_TICK_LENGTH
            
            # Draw tick
            start_point = (self.width - self.PITCH_WIDTH, y_pos)
            end_point = (self.width - self.PITCH_WIDTH + tick_length, y_pos)
            cv2.line(img, start_point, end_point, self.WHITE, 1, cv2.LINE_AA)

    def update(self, img, pitch, yaw):
        # Create blank image
        # img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw indicators
        self.draw_yaw_indicator(img, yaw)
        self.draw_pitch_indicator(img, pitch)
        
        return img





# Example usage
if __name__ == "__main__":
    print("Importing...")
    import cv2
    import time
    
    # Initialize framebuffer
    print("Framebuffer...")
    fb = FrameBuffer()

    hud = PitchYawHUD(width=fb.xres, height=fb.yres)
    
    # Create a test pattern
    print("Creating test pattern...")
    test_frame = np.zeros((fb.yres, fb.xres, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fb.xres)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fb.yres)
    time.sleep(2)


    # cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, fb.xres)
    # cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, fb.yres)
    # cap.set(cv2.CV_CAP_PROP_EXPOSURE, 0.1)

    while True:
        ret, frame = cap.read()
        if not ret:
            # print("err") 
            continue

        print(frame.shape)

        # frame = frame[:fb.xres, :fb.yres]
        frame = cv2.resize(frame, dsize=(fb.xres, fb.yres), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # test_frame[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
        
        frame = hud.update(frame, 0, 0)

        # fb.display_frame(frame)
        cv2.imshow("e", frame)
        cv2.waitKey(1)

# convert "/usr/share/rpd-wallpaper/raspberry-pi-logo.png"\["$fbw"x"$fbh"^\] +flip -strip -define bmp:subtype=RGB565 bmp2:- | tail -c $(( fbw * fbh * fbd / 8 )) > /dev/fb0