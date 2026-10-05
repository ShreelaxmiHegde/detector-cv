import cv2
import sys
import mediapipe as mp
from mediapipe.tasks.python import vision
from helper import draw, draw_landmarks

s = 0
if len(sys.argv) > 1:
  s = sys.argv[1]

win_name = 'Hand Landmark Capture'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(s)

VisionRunningMode = vision.RunningMode
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerResult = vision.GestureRecognizerResult
GestureRecognizerOptions = vision.GestureRecognizerOptions
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task')

# This will contain the latest MediaPipe result
latest_result = None

def result_callback(
  detection_result: GestureRecognizerResult,
  output_image: mp.Image,
  timestamp_ms: int
):
  global latest_result
  latest_result = detection_result

options = GestureRecognizerOptions(
  base_options=base_options, 
  num_hands=2, 
  running_mode=VisionRunningMode.LIVE_STREAM,
  result_callback=result_callback
)
recognizer = GestureRecognizer.create_from_options(options)

frame_timestamp_ms = 0
prev_gesture = None
overlay = cv2.imread("assets/overlay.png")

while cv2.waitKey(1) != 27:
  has_frame, frame = cap.read()
  if not has_frame:
    print("Could not read camera frame")
    break

  frame = cv2.flip(frame, 1)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  recognizer.recognize_async(mp_img, frame_timestamp_ms)

  curr_gesture = None
  if latest_result is not None:
    for i, gesture in enumerate(latest_result.gestures):
      curr_gesture = gesture[0].category_name
      break

    if prev_gesture != 'Open_Palm' and curr_gesture == 'Open_Palm':
      print('detected open palm ', frame_timestamp_ms)
      draw(overlay)

    prev_gesture = curr_gesture

    # Draw landmarks
    draw_landmarks(frame_rgb, latest_result.hand_landmarks)
    
  display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
  overlay = cv2.resize(overlay, (display_frame.shape[1], display_frame.shape[0]))

  # Display
  output = cv2.addWeighted(
    display_frame, 0.2,
    overlay, 0.8,
    0
  )

  cv2.imshow(win_name, output)

  frame_timestamp_ms += 30

cap.release()
cv2.destroyAllWindows()