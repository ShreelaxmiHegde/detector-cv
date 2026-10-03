import cv2
import numpy as np
import random

FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_shapes_tip_coordinates(
  frame,
  x_coordinates: list,
  y_coordinates: list
):
  height, width, _ = frame.shape

  thumb_point_x = int(x_coordinates[4]*width)
  thumb_point_y = int(y_coordinates[4]*height)
  
  index_point_x = int(x_coordinates[8]*width)
  index_point_y = int(y_coordinates[8]*height)
  
  p1 = np.array([thumb_point_x, thumb_point_y])
  p2 = np.array([index_point_x, index_point_y])
  
  # distance between thumb and index finger tips
  dist = int(np.linalg.norm(p1 - p2))
  print(dist)
  
  cv2.circle(frame, (thumb_point_x, thumb_point_y), int(dist/2), (230, 210, 72), 2)

def show_ref_hand_lines(
  frame,
  x_coordinates: list,
  y_coordinates: list,
  handedness
):
  height, width, _ = frame.shape

  text_x = int(min(x_coordinates) * width)
  text_y = int(min(y_coordinates) * height) - 10
  
  cv2.putText(
    frame,
    handedness[0].category_name,
    (text_x, text_y),
    cv2.FONT_HERSHEY_DUPLEX,
    FONT_SIZE,
    HANDEDNESS_TEXT_COLOR,
    FONT_THICKNESS,
    cv2.LINE_AA
  )

def draw(
  frame
):
  height, width, _ = frame.shape
  pt1 = (random.randint(0, 255)*height-10, random.randint(0, 255)*width)
  pt2 = (random.randint(0, 255)*height-10, random.randint(0, 255)*width)
  print(pt1, pt2)
  cv2.rectangle(frame, (20, 20), pt1, pt2, 2, cv2.LINE_AA)

def find_angles(frame, x_coordinates, y_coordinates):
  height, width, _ = frame.shape

  x1 = int(x_coordinates[2]*width)
  y1 = int(y_coordinates[2]*height)

  x2 = int(x_coordinates[5]*width)
  y2 = int(y_coordinates[5]*height)

  dx = np.array(x1 - x2)
  dy = np.array(y1 - y2)

  print(dx, dy)

  angle_deg = np.rad2deg(np.arctan2(dy, dx))
  print(angle_deg)