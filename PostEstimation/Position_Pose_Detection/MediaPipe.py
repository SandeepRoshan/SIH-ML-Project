import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, draw=False, display=False):
    '''
    Perform pose detection on an image.
    Args:
        image: Input image with a person.
        pose: Pose function for detection.
        draw: Whether to draw landmarks on the image.
        display: Whether to show images using matplotlib.
    Returns:
        output_image: Image with landmarks drawn if specified.
        results: Pose landmarks detection results.
    '''
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    
    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(
            image=output_image, 
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237), thickness=2, circle_radius=2)
        )
        
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121); plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off');
        plt.subplot(122); plt.imshow(output_image[:,:,::-1]); plt.title("Output Image"); plt.axis('off');
    else:
        return output_image, results
    
def checkHandsJoined(image, results, draw=False, display=False):
    '''
    Check if hands are joined in an image.
    Args:
        image: Input image with a person.
        results: Pose landmarks detection results.
        draw: Whether to draw status on the image.
        display: Whether to show image using matplotlib.
    Returns:
        output_image: Image with status written if specified.
        hand_status: Status of hands (joined or not).
    '''
    height, width, _ = image.shape
    output_image = image.copy()
    
    left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    distance = int(hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1]))
    
    if distance < 130:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)
    else:
        hand_status = 'Hands Not Joined'
        color = (0, 0, 255)
        
    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {distance}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
    
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]); plt.title("Output Image"); plt.axis('off');
    else:
        return output_image, hand_status

try:
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)
    cv2.namedWindow('Hands Joined?', cv2.WINDOW_NORMAL)
    
    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        frame, results = detectPose(frame, pose_video, draw=True)
        
        if results.pose_landmarks:
            frame, _ = checkHandsJoined(frame, results, draw=True)
        
        cv2.imshow('Hands Joined?', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    camera_video.release()
    cv2.destroyAllWindows()