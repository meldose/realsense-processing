import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Align depth to color
align = rs.align(rs.stream.color)
pipeline.start(config)

# Get depth scale and camera intrinsics
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_profile = profile.get_stream(rs.stream.color)
intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
print(f"[INFO] fx: {fx}, fy: {fy}, depth_scale: {depth_scale}")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Preprocess color image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional: morphological close to clean small holes
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 50000:
                continue  # Ignore too small or large contours

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mark the center of the box
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)

            # Extract depth ROI and filter invalid values
            roi_depth = depth_image[y:y + h, x:x + w].astype(float) * depth_scale
            roi_depth = roi_depth[(roi_depth > 0.1) & (roi_depth < 2.0)]

            if roi_depth.size > 0:
                median_depth = np.median(roi_depth)

                width_m = (w * median_depth) / fx
                height_m = (h * median_depth) / fy
                depth_m = median_depth  # Z-direction

                # Display live length and breadth in the image
                dimensions_text = f"L: {width_m:.2f}m, B: {height_m:.2f}m"
                cv2.putText(color_image, dimensions_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Optional: display depth (distance from camera)
                depth_text = f"H: {depth_m:.2f}m"
                cv2.putText(color_image, depth_text, (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                print("[WARNING] No valid depth in ROI")

        # Display combined image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        combined = np.hstack((color_image, depth_colormap))

        # Show image with live measurements
        cv2.imshow("RealSense Live Object Measurement", combined)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
