# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

import cv2
import mediapipe as mp
import time
import math
import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download

from pgr_mlp import PointerMLP

class HandController:
    def __init__(self, width=600, height=500, control_threshold=0.9999):
        self.control_mode = False
        self.control_threshold = control_threshold

        self.repo_id = "TheRealAppleBoi/pointer_gesture_recognizer"
        self.model_filename = "pointer_model.pth"
        self.local_dir = "./models"
        os.makedirs(self.local_dir, exist_ok=True)

        self.pgr = self.load_model()

        self.window_titles = ["Hand Capture", "Virtual Buttons"]
        self.button_pressed = "Center"
        self.buttons_config = {"Left": [None, (255, 0, 0)], "Center": [None, (0, 255, 0)], "Right": [None, (0, 0, 255)], "Jump": [None, (255, 255, 0)], "Slide": [None, (255, 0, 255)]}

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hand = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.2,
            static_image_mode=False
        )

        self.finger_tips = [4, 8, 12, 16, 20] # thumb, index, middle, ring, pinky
        self.finger_mcps = [2, 5, 9, 13, 17]
        self.finger_ips = [3, 7, 11, 15, 19]
        self.wrist = 0
    
    def load_model(self):
        local_model_path = os.path.join(self.local_dir, self.model_filename)
        if not os.path.exists(local_model_path):
            print("Downloading model from Hugging Face...")
            local_model_path = hf_hub_download(repo_id=self.repo_id, 
                                               filename=self.model_filename, 
                                               local_dir=self.local_dir,
                                               repo_type="model")
            print("Downloaded to:", local_model_path)
        else:
            print("Model already exists locally:", local_model_path)

        model = PointerMLP(input_size=21)
        model.load_state_dict(torch.load(local_model_path))
        model.eval()
        print("Model loaded successfully!")
        return model
    
    def distance(self, x1, x2, y1, y2, z1=None, z2=None):
        if z1!=None and z2!=None:
            return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        return math.sqrt((x1-x2)**2+(y1-y2)**2)

    def check_control(self, landmark):
        """return True when making a pointer"""
        wrist = landmark[self.wrist]
        distances = []
        for lm in landmark:
            dist = self.distance(lm.x, wrist.x, lm.y, wrist.y, lm.z, wrist.z)
            distances.append(dist)

        if self.pgr is not None:
            output = self.pgr(torch.tensor(distances, dtype=torch.float32).unsqueeze(0))
            pred = (output > self.control_threshold).int().item()
            return bool(pred), output
        return False, None
    
    def find_control_point(self, landmark, frame):
        cx, cy = landmark[self.finger_tips[1]].x, landmark[self.finger_tips[1]].y 
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = int(cx*w), int(cy*h)
        cv2.circle(frame, (x, y), 20, (255,0,255), -1)
        return cx, cy
    
    def draw_info(self, frame, text, pos=(20, 40)):
        cv2.putText(frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def draw_buttons(self, frame):
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay = frame.copy()

        btn_width = w // 3
        remainder = w % 3 
        widths = [btn_width, btn_width, btn_width]
        for i in range(remainder):
            widths[i] += 1 

        # Full height split: top (jump), middle (3 buttons), bottom (slide)
        top_height = h // 6
        bottom_height = h // 6
        center_height = h - top_height - bottom_height  # Exact middle height

        # Starting Y positions
        y_jump = 0
        y_center = top_height
        y_slide = h - bottom_height

        # === Jump (top full-width rectangle) ===
        jump_rect = (0, y_jump, w, y_center)
        self.buttons_config["Jump"][0] = jump_rect
        cv2.rectangle(overlay, (0, y_jump), (w, y_center), self.buttons_config["Jump"][1], -1)

        # === Slide (bottom full-width rectangle) ===
        slide_rect = (0, y_slide, w, h)
        self.buttons_config["Slide"][0] = slide_rect
        cv2.rectangle(overlay, (0, y_slide), (w, h), self.buttons_config["Slide"][1], -1)

        # === Left, Center, Right (middle row) ===
        x_offset = 0
        for i, name in enumerate(["Left", "Center", "Right"]):
            x1 = x_offset
            x2 = x_offset + widths[i]
            rect = (x1, y_center, x2, y_center + center_height)
            self.buttons_config[name][0] = rect
            cv2.rectangle(overlay, (x1, y_center), (x2, y_center + center_height), self.buttons_config[name][1], -1)
            x_offset = x2  # Next button starts exactly where this one ends

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for name, config in self.buttons_config.items():
            (x1, y1, x2, y2) = config[0]
            color = config[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Center text in button
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(frame, name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def run(self):
        while True:
            success, frame = self.cap.read() # Frames in BGR
            if not success:
                break

            flipped_frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            result = self.hand.process(rgb_frame)

            hand_frame = flipped_frame.copy()
            button_frame = flipped_frame.copy()

            self.draw_buttons(button_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                # print(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    hand_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS, 
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                    )

                control_mode, conf = self.check_control(hand_landmarks.landmark)
                mode_text = f"Pointer (Control Mode)" if control_mode else "Open (Idle)"
                self.draw_info(hand_frame, mode_text)
                if conf:
                    self.draw_info(hand_frame, f"{float(conf):.4f}", (20, 80))

                if control_mode:
                    cx, cy = self.find_control_point(hand_landmarks.landmark, hand_frame)
                    w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    for name, config in self.buttons_config.items():
                        if cx > (config[0][0]/w) and cx < (config[0][2]/w) and cy > (config[0][1]/h) and cy < (config[0][3]/h):
                            self.button_pressed = name

                    match self.button_pressed:
                        case "Center":
                            self.middle_lane()
                        case "Left":
                            self.left_lane()
                        case "Right":
                            self.right_lane()
                        case 'Jump':
                            self.jump()
                        case "Slide":
                            self.slide()
                    time.sleep(0.1)

            cv2.imshow(self.window_titles[0], hand_frame)
            cv2.imshow(self.window_titles[1], button_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def jump(self):
        print("JUMP!")
        pass

    def slide(self):
        print("SLIDE!")
        pass

    def left_lane(self):
        print("GO LEFT!")
        pass

    def right_lane(self):
        print("GO RIGHT!")
        pass

    def middle_lane(self):
        print("GO CENTER")
        pass



if __name__ == "__main__":
    controller = HandController()
    controller.run()