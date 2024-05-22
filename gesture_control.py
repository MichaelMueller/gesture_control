import argparse
import time
import json
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

@dataclass
class Config:
    cam = 0
    model_complexity = 1
    smoothing_factor = 0.3
    ratio_tips_closed = 0.19
    ratio_tips_closing = 0.7
    max_consec_frames_for_click = 7
    min_detection_confidence=0.66
    min_tracking_confidence=0.66
    simulate_click_by_mouse_down_and_up = True
    max_output_win_size = 400
    frame_resize_to_width = None
    app_name = "Gesture Control"
    
# Variables to store smoothed coordinates
pyautogui.FAILSAFE = False  # Disables the fail-safe
config = Config()
last_x, last_y = None, None
consecutive_frames = 0
mouse_down = False
window_opened = False
last_button = None

def move_mouse(wrist_coords, smoothing_factor):
    global last_x, last_y
    
    if last_x is None and last_y is None:
        last_x = wrist_coords[0]
        last_y = wrist_coords[1]
        return
    
    x_offset = ( wrist_coords[0] - last_x ) * smoothing_factor
    y_offset = ( wrist_coords[1] - last_y ) * smoothing_factor
    if interaction_enabled:
        pyautogui.moveRel( x_offset, y_offset, duration=0, _pause=False )
        
import cv2

def is_point_inside_image(image, point):
    # Image dimensions
    height, width = image.shape[:2]

    # Point coordinates
    x, y = point

    # Check if the point is within the bounds of the image
    return 0 <= x < width and 0 <= y < height

def find_finger_tips(frame, hand_landmarks, handedness ):
    tips_together = False
    tips_closing = False
    wrist_coords = None
    ratio_index_thumb_wrist = None
    ratio_middle_thumb_wrist = None
    button = "left"
    
    if hand_landmarks:
        wrist = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
        index_tip = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        h, w, _ = frame.shape
        wrist_coords = (int(wrist.x * w), int(wrist.y * h))
        index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
        thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        middle_finger_tip_coords = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
        
        palm_facing = thumb_tip_coords[0] < index_tip_coords[0] if handedness[0].classification[0].label == 'Right' else thumb_tip_coords[0] > index_tip_coords[0]
        if is_point_inside_image( frame, wrist_coords ) and palm_facing == True:            
            cv2.circle(frame, wrist_coords, 10, (0, 0, 255), -1)
            cv2.circle(frame, index_tip_coords, 10, (255, 255, 0), -1)
            cv2.circle(frame, thumb_tip_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, middle_finger_tip_coords, 10, (0, 255, 255), -1)

            distance_wrist_thumb = np.linalg.norm(np.array(thumb_tip_coords) - np.array(wrist_coords))
            distance_index_thumb = np.linalg.norm(np.array(index_tip_coords) - np.array(thumb_tip_coords))
            distance_middle_finger_thumb = np.linalg.norm(np.array(middle_finger_tip_coords) - np.array(thumb_tip_coords))
            
            ratio_index_thumb_wrist = distance_index_thumb / distance_wrist_thumb
            ratio_middle_thumb_wrist = distance_middle_finger_thumb / distance_wrist_thumb
            buttons = ["left", "right"]
            
            for idx, curr_ratio in enumerate([ratio_index_thumb_wrist, ratio_middle_thumb_wrist]):       
                tips_together = curr_ratio < config.ratio_tips_closed
                if tips_closing == False:
                    tips_closing = tips_together == False and curr_ratio < config.ratio_tips_closing
                    button = buttons[idx]
                    
                if tips_together:
                    button = buttons[idx]
                    break
                    
            
            #cv2.putText(frame, f'index<->thumb: {distance:.2f}, thumb<->wrist: {distance_wrist_thumb:.2f}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, f'tips_together: {tips_together}, tips_closing: {tips_closing}, min_max: {global_ratios[0]:.2f}, {global_ratios[1]:.2f}, ratio: {ratio:.2f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            wrist_coords = None

    return frame, tips_together, wrist_coords, tips_closing, ratio_index_thumb_wrist, ratio_middle_thumb_wrist, button

def process_frame(frame, hands, smoothing_factor, start_time, num_frames, avg_fps):
    global consecutive_frames, mouse_down, last_x, last_y, last_button
    if config.frame_resize_to_width != None:
        print(f"resizing to {config.frame_resize_to_width}")
        frame = resize_keep_aspect_ratio( frame, config.frame_resize_to_width )
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tips_together = False
    tips_closing = False
    wrist_coords = None
    ratio_index_thumb_wrist = None
    ratio_middle_thumb_wrist = None
    button = None

    if results.multi_hand_landmarks:
        frame, tips_together, wrist_coords, tips_closing, ratio_index_thumb_wrist, ratio_middle_thumb_wrist, button = find_finger_tips(frame, results.multi_hand_landmarks, results.multi_handedness)
        # only move if we have coords and we are in the closing state or in a mouse_down state
        if wrist_coords != None and ( tips_closing == False or mouse_down == True ):
            move_mouse(wrist_coords, smoothing_factor)
        else:
            last_x = None
            last_y = None
            
        if interaction_enabled:
            if tips_together:
                consecutive_frames += 1
                if mouse_down == False and consecutive_frames >= config.max_consec_frames_for_click + 1:
                    mouse_down = True
                    last_button = button
                    pyautogui.mouseDown(button=button, _pause=False)
                    print("mouseDown")
                    
            else:
                if mouse_down == True:
                    if tips_closing == False:
                        mouse_down = False
                        print("mouseUp")
                        pyautogui.mouseUp(button=last_button, _pause=False)
                elif consecutive_frames >= 1:
                    if config.simulate_click_by_mouse_down_and_up:
                        pyautogui.mouseDown(button=button, _pause=True)
                        pyautogui.mouseUp(button=button, _pause=False)
                    else:
                        pyautogui.click(button=button, _pause=False)
                    print("click")
                    
                consecutive_frames = 0
            
    # Calculate FPS
    num_frames += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    current_fps = num_frames / elapsed_time
    avg_fps = (avg_fps * (num_frames - 1) + current_fps) / num_frames

    # Print FPS on frame
    interaction_enabled_txt = "Interaction enabled" if interaction_enabled else "Interaction disabled"
    measurements_text = f'{interaction_enabled_txt}, FPS: {current_fps:.2f}'    
    if ratio_index_thumb_wrist:
        measurements_text +=  f', index tip: {ratio_index_thumb_wrist:.2f}'
    if ratio_middle_thumb_wrist:
        measurements_text +=  f', middle tip: {ratio_middle_thumb_wrist:.2f}'
    font_size = 0.75
    cv2.putText(frame, measurements_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), 2, cv2.LINE_AA)
    
    if button != None:
        state = "No Hand detected"
        if wrist_coords:
            state = "Mouse moving"
        if tips_together:
            state = "Tips closed"
        elif tips_closing:
            state = "Locked"
        cv2.putText(frame, f'State: {state}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2, cv2.LINE_AA)
        if mouse_down is True:
            cv2.putText(frame, f'{button} Mouse pressed!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2, cv2.LINE_AA)
    return frame, tips_together, num_frames, avg_fps

def resize_keep_aspect_ratio(image, width):
    # Calculate the ratio of the new width to the old width
    ratio = width / image.shape[1]
    
    # Calculate the new height based on the ratio
    height = int(image.shape[0] * ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (width, height))
    
    return resized_image

def move_window_to_lower_left(window_name, window_height):
    # Get screen dimensions using pyautogui
    _, screen_height = pyautogui.size()

    # Calculate the new position
    x_position = 0  # Lower-left corner
    y_position = screen_height - window_height - 40 - 40# Lower-left corner minus taskbar
    
    # Move the OpenCV window to the specified position
    cv2.moveWindow(config.app_name, x_position, y_position)

interaction_enabled = False

def main():
    global window_opened, config, interaction_enabled
    parser = argparse.ArgumentParser(description=config.app_name)
    parser.add_argument("--config", type=str, default=None, help="Location of a config file")
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, "r") as f:
            dict_:dict = json.load(f)
            for key, value in dict_.items():
                config.__dict__[key] = value

    cap = cv2.VideoCapture(config.cam)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence
    )

    start_time = time.time()
    num_frames = 0
    avg_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream.")
            break

        frame, tips_together, num_frames, avg_fps = process_frame(frame, hands, config.smoothing_factor, start_time, num_frames, avg_fps)

        frame = resize_keep_aspect_ratio(frame, config.max_output_win_size)
        cv2.imshow(config.app_name, frame)
        if not window_opened:
            window_opened = True
            cv2.setWindowProperty(config.app_name, cv2.WND_PROP_TOPMOST, 1)   
            move_window_to_lower_left(config.app_name, frame.shape[0])         

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            interaction_enabled = not interaction_enabled
        

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
