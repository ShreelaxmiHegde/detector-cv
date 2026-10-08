- Google Mediapipe hand landmark detection implementation for images. 

- Used opencv for hand landmark detection in live streaming.

- Extracting finger tip points - drawing and manipulating shapes on camera frame.

- Used mediapipe methods to get finger tip coordinates and calculated the distance between 2 finger tips.

- Used that distance to draw shapes(circle, straight lines) using opencv functions for shapes.

        - Got the result somewhere else than finger tip positions because of the coordinate points meaning mismatch in Mediapipe and opencv.

        - Used the frame height and width to define the units.

- but, as live stream generates frames frequently, the drawn shapes also get generated each time. 

        - used a static overlay for gesture based manipulations and use the camera frame only to detect gestures.

- Mediapipe gesture-detector model limitation to only a few gestures:

         - Finding better ways:- (No ML model involvement. Ex: distance, angles)

         - 1. Angle between 2 points (thumb_mcp, index_mcp) : camera coordinate and hand intrinsic property reference problem. (different hand rotation/position/distance from camera give inconsistent angles)

         - 2. use mediapipe gesture detector for basic gestures