import cv2
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
overlay = cv2.imread("../rocket.jpg")

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

      # Bounding-box text position
      height, width, _ = frame_rgb.shape

      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]

      thumb_point_x = int(x_coordinates[4]*width)
      thumb_point_y = int(y_coordinates[4]*height)

      index_point_x = int(x_coordinates[8]*width)
      index_point_y = int(y_coordinates[8]*height)

      p1 = np.array([thumb_point_x, thumb_point_y])
      p2 = np.array([index_point_x, index_point_y])

      # distance between thumb and index finger tips
      dist = int(np.linalg.norm(p1 - p2))
      print(x_coordinates[4], y_coordinates[4])
      print(thumb_point_x, thumb_point_y)

      cv2.circle(frame_rgb, (thumb_point_x, thumb_point_y), int(dist/2), (230, 210, 72), 2)
      # cv2.arrowedLine(frame_rgb, (20, 20), (int(thumb_point_x*100), int(thumb_point_y*100)), (200, 100, 250), 1, cv2.LINE_AA)

      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - 10

      cv2.putText(
        frame_rgb,
        handedness[0].category_name,
        (text_x, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        FONT_SIZE,
        (88, 205, 54),
        FONT_THICKNESS,
        cv2.LINE_AA
      )

  display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
  overlay = cv2.resize(overlay, (display_frame.shape[1], display_frame.shape[0]))

  # Display
  output = cv2.addWeighted(
    display_frame, 1.0,
    overlay, 0.0,
    0
  )

  cv2.imshow(win_name, output)

  frame_timestamp_ms += 33

cap.release()
cv2.destroyAllWindows()