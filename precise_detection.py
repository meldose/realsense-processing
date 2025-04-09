import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque

# Initialize pipeline and config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Depth scale and smoothing buffer
depth_scale = 0.001
depth_history = deque(maxlen=5)

# Align depth to color
align = rs.align(rs.stream.color)

# RealSense depth filters
dec_filter = rs.decimation_filter()      # Reduces resolution to remove noise
spat_filter = rs.spatial_filter()        # Smooths spatially
temp_filter = rs.temporal_filter()       # Reduces temporal noise
hole_filter = rs.hole_filling_filter()   # Fills small holes in depth

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Apply depth filters
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = spat_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_frame = hole_filter.process(depth_frame)

        # Get data
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)

        # Improve segmentation using Otsu + morphology
        blurred = cv2.GaussianBlur(depth_vis, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                depth_crop = depth_image[y:y+h, x:x+w]
                valid_depths = depth_crop[depth_crop > 0]

                if valid_depths.size > 0:
                    # Robust height with percentiles
                    depth_10 = np.percentile(valid_depths, 10) * depth_scale
                    depth_90 = np.percentile(valid_depths, 90) * depth_scale
                    object_depth = depth_90 - depth_10

                    depth_history.append(object_depth)
                    smoothed_height = np.mean(depth_history)

                    # Real-world size via deprojection
                    median_depth = np.median(valid_depths) * depth_scale
                    p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], median_depth)
                    p2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x+w, y+h], median_depth)

                    length = abs(p2[0] - p1[0])
                    width = abs(p2[1] - p1[1])

                    # Draw results
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0,255,0), 2)
                    label = f"L: {length:.2f}m, W: {width:.2f}m, H: {smoothed_height:.2f}m"
                    cv2.putText(color_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Show visuals
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
