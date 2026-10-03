import cv2
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from helper import find_angles

s = 0
if len(sys.argv) > 1:
  s = sys.argv[1]

win_name = 'Hand Landmark Capture'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(s)

MARGIN = 10   # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# This will contain the latest MediaPipe result
latest_result = None

def result_callback(
  detection_result: HandLandmarkerResult,
  output_image: mp.Image,
  timestamp_ms: int
):
  global latest_result
  latest_result = detection_result


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
  base_options=base_options, 
  num_hands=2, 
  running_mode=VisionRunningMode.LIVE_STREAM,
  result_callback=result_callback
)
detector = vision.HandLandmarker.create_from_options(options)

frame_timestamp_ms = 0
overlay = cv2.imread("assets/overlay.png")

while cv2.waitKey(1) != 27:
  has_frame, frame = cap.read()
  if not has_frame:
    print("Could not read camera frame")
    break

  frame = cv2.flip(frame, 1)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  detector.detect_async(mp_img, frame_timestamp_ms)

  if latest_result is not None:
    for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
      handedness = latest_result.handedness[idx]

      # Draw landmarks
      mp_drawing.draw_landmarks(
        frame_rgb,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
      )

      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]

      find_angles(frame_rgb, x_coordinates, y_coordinates)

  display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
  overlay = cv2.resize(overlay, (display_frame.shape[1], display_frame.shape[0]))

  # Display
  output = cv2.addWeighted(
    display_frame, 0.2,
    overlay, 0.8,
    0
  )

  cv2.imshow(win_name, output)

  frame_timestamp_ms += 33

cap.release()
cv2.destroyAllWindows()