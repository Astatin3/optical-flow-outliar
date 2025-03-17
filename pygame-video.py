from pygame.locals import KEYDOWN, K_ESCAPE, K_q
import pygame
import cv2
import sys

size = [400, 300]

camera = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode(size)

font = pygame.font.SysFont('Arial', 12)

MAJOR_TICK_LENGTH = 15
MINOR_TICK_LENGTH = 8
LABEL_OFFSET = 20

WHITE = (255, 255, 255)
GRAY = (128, 128, 128)


YAW_HEIGHT = 1
PITCH_WIDTH = 10

class PitchYawHUD:
    def __init__(self, screen_width=800, screen_height=600):
        # pygame.init()
        # screen = pygame.display.set_mode((screen_width, screen_height))
        # pygame.display.set_caption("Pitch & Yaw HUD")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        
        # HUD dimensions
        self.YAW_HEIGHT = 1
        self.PITCH_WIDTH = 10
        
        # Tick marks
        self.MAJOR_TICK_LENGTH = 15
        self.MINOR_TICK_LENGTH = 8
        self.LABEL_OFFSET = 20
        
        # Font
        self.font = pygame.font.SysFont('Arial', 12)
        
        # Cardinal directions for yaw
        self.cardinal_directions = {
            0: "N", 45: "NE", 90: "E", 135: "SE",
            180: "S", 225: "SW", 270: "W", 315: "NW"
        }
    
    def draw_yaw_indicator(self, screen, yaw_angle):
        # Normalize yaw angle to 0-360
        yaw_angle = yaw_angle % 360
        
        
        # Calculate pixel per degree for yaw
        pixels_per_degree = screen.get_width() / 360
        
        # Draw tick marks
        for angle in range(0, 360, 5):  # Draw every 5 degrees
            x_pos = (angle - yaw_angle) * pixels_per_degree
            x_pos = x_pos % screen.get_width()
            
            # Determine tick length
            if angle % 45 == 0:  # Cardinal and intercardinal directions
                tick_length = self.MAJOR_TICK_LENGTH
                # Draw direction label
                if angle in self.cardinal_directions:
                    label = self.font.render(self.cardinal_directions[angle], True, self.WHITE)
                    screen.blit(label, (x_pos - label.get_width()/2, self.LABEL_OFFSET))
            else:
                tick_length = self.MINOR_TICK_LENGTH
            
            # Draw tick
            pygame.draw.line(screen, self.WHITE,
                           (x_pos, 0),
                           (x_pos, tick_length))
    
    def draw_pitch_indicator(self, screen,  pitch_angle):
        # Normalize pitch angle to -180 to 180
        pitch_angle = max(-180, min(180, pitch_angle))
        
        # Draw background
        # pygame.draw.rect(screen, self.GRAY,
        #                 (screen.get_width() - self.PITCH_WIDTH, 0,
        #                  self.PITCH_WIDTH, screen.get_height()))
        
        # Calculate pixel per degree for pitch
        pixels_per_degree = screen.get_height() / 360
        
        # Draw tick marks
        for angle in range(-180, 181, 10):  # Draw every 10 degrees
            y_pos = screen.get_height()/2 + (angle - pitch_angle) * pixels_per_degree
            
            # Skip if outside screen
            if y_pos < 0 or y_pos > screen.get_height():
                continue
            
            # Determine tick length
            if angle % 30 == 0:  # Major ticks
                tick_length = self.MAJOR_TICK_LENGTH
                # Draw angle label
                label = self.font.render(str(angle), True, self.WHITE)
                screen.blit(label,
                               (screen.get_width() - self.PITCH_WIDTH - label.get_width() - 5,
                                y_pos - label.get_height()/2))
            else:
                tick_length = self.MINOR_TICK_LENGTH
            
            # Draw tick
            pygame.draw.line(screen, self.WHITE,
                           (screen.get_width() - self.PITCH_WIDTH, y_pos),
                           (screen.get_width() - self.PITCH_WIDTH + tick_length, y_pos))

    def update(self, screen, pitch, yaw):
        # screen.fill((0, 0, 0))  # Clear screen
        self.draw_yaw_indicator(screen, yaw)
        self.draw_pitch_indicator(screen, pitch)
        pygame.display.flip()



hud = PitchYawHUD()

pitch = 0
yaw = 0



try:
    while True:

        ret, frame = camera.read()

        if not ret: continue

        screen.fill([0, 0, 0])
        frame = cv2.resize(frame, dsize=(size[0], size[1]), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()

        hud.update(screen, pitch, yaw)

        pitch += 1
        yaw += 1


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    sys.exit(0)

except (KeyboardInterrupt, SystemExit):
    pygame.quit()
    cv2.destroyAllWindows()