# import pyrealsense2 as rs # imported module pyrelease
# import numpy as np # imported numpy module
# import cv2 # imported cv2 module

# # Set up the RealSense pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # Start streaming
# pipeline.start(config)

# # Create an align object to align depth and color frames
# align_to = rs.stream.color
# align = rs.align(align_to)

# # Get frame size
# frame_width = 640
# frame_height = 480

# try:
#     while True:
#         # Wait for the next set of frames
#         frames = pipeline.wait_for_frames()

#         # Align depth frame to color frame (optional)
#         aligned_frames = align.process(frames)
#         depth_frame = aligned_frames.get_depth_frame()

#         # Convert depth frame to numpy array
#         depth_image = np.asanyarray(depth_frame.get_data())

#         # Normalize the depth image to 8-bit for contour detection
#         depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#         # Apply colorization to depth frame for visualization (optional)
#         colorizer = rs.colorizer()
#         depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

#         # Threshold or detect the object in the depth frame
#         _, thresholded_depth = cv2.threshold(depth_image_normalized, 100, 255, cv2.THRESH_BINARY)

#         # Find contours of the object in the thresholded image
#         contours, _ = cv2.findContours(thresholded_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Filter small contours
#                 # Get the bounding box of the object
#                 x, y, w, h = cv2.boundingRect(contour)

#                 # Ensure that the coordinates are within valid bounds
#                 x = min(x, frame_width - 1)
#                 y = min(y, frame_height - 1)
#                 x2 = min(x + w, frame_width - 1)
#                 y2 = min(y + h, frame_height - 1)

#                 # Measure the object in 3D space using depth data (center of bounding box)
#                 object_depth_top_left = depth_frame.get_distance(x, y)
#                 object_depth_bottom_right = depth_frame.get_distance(x2, y2)

#                 # Calculate the real-world length and breadth (in meters)
#                 length = (object_depth_bottom_right - object_depth_top_left) * w
#                 breadth = (object_depth_bottom_right - object_depth_top_left) * h

#                 # Get the object's height (you can take the average of the depths)
#                 avg_depth = np.mean(depth_image[y:y+h, x:x+w])  # Average depth of the bounding box area
#                 height = avg_depth  # Height corresponds to depth

#                 # Print or display the object length, breadth, and height
#                 print(f"Object Length: {length:.2f} meters")
#                 print(f"Object Breadth: {breadth:.2f} meters")
#                 print(f"Object Height (Depth): {height:.2f} meters")

#                 # Draw the bounding box and show the image
#                 cv2.rectangle(depth_colormap, (x, y), (x2, y2), (0, 255, 0), 2)

#         # Show the depth image
#         cv2.imshow("Depth Image", depth_colormap)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     # Stop the pipeline
#     pipeline.stop()
#     cv2.destroyAllWindows()

import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

depth_scale = 0.001  # typical for D435, 1mm per unit

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize for visualization
        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)

        # Threshold depth to isolate object (tune values as needed)
        _, thresh = cv2.threshold(depth_vis, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                depth_crop = depth_image[y:y+h, x:x+w]
                valid_depths = depth_crop[depth_crop > 0]

                if valid_depths.size > 0:
                    min_depth = np.min(valid_depths) * depth_scale
                    max_depth = np.max(valid_depths) * depth_scale
                    object_depth = max_depth - min_depth  # height

                    length = w * depth_scale
                    width = h * depth_scale

                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0,255,0), 2)
                    label = f"L: {length:.2f}m, W: {width:.2f}m, H: {object_depth:.2f}m"
                    cv2.putText(color_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

