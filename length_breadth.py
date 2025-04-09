import pyrealsense2 as rs # imorted the module pyrealsense2
import numpy as np # imported numpy module
import cv2 # impored cv2 module
from collections import deque # imported collections module

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

depth_scale = 0.001  # typical for D435 (1mm per unit)
depth_history = deque(maxlen=5)  # For smoothing height

# Align depth to color stream
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)

        # Threshold for segmentation
        _, thresh = cv2.threshold(depth_vis, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                depth_crop = depth_image[y:y+h, x:x+w]
                valid_depths = depth_crop[depth_crop > 0]

                if valid_depths.size > 0:
                    # Robust height estimation
                    depth_10 = np.percentile(valid_depths, 10) * depth_scale
                    depth_90 = np.percentile(valid_depths, 90) * depth_scale
                    object_depth = depth_90 - depth_10

                    # Smooth height over frames
                    depth_history.append(object_depth)
                    smoothed_height = np.mean(depth_history)

                    # Use deprojection to get real-world size
                    median_depth = np.median(valid_depths) * depth_scale
                    p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], median_depth)
                    p2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x+w, y+h], median_depth)
                    length = abs(p2[0] - p1[0])
                    width = abs(p2[1] - p1[1])

                    # Calculate the center of the bounding box
                    cx = x + w // 2
                    cy = y + h // 2
                    center_depth = depth_image[cy, cx] * depth_scale

                    # Fallback if center depth is zero (invalid pixel)
                    if center_depth == 0:
                        non_zero_depths = depth_crop[depth_crop > 0]
                        center_depth = np.median(non_zero_depths) * depth_scale

                    # Deproject to 3D space (X, Y, Z in meters)
                    object_center_3D = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], center_depth)
                    X, Y, Z = object_center_3D  # in meters

                    # Draw and label
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0,255,0), 2)
                    label = f"X: {X:.2f}m, Y: {Y:.2f}m, Z: {Z:.2f}m\nL: {length:.2f}m, W: {width:.2f}m, H: {smoothed_height:.2f}m"
                    cv2.putText(color_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop() # stopping the pipeline
    cv2.destroyAllWindows() # closing all the windows
