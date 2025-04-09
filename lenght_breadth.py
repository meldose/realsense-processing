import pyrealsense2 as rs
import numpy as np
import cv2

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

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue  # ignore small noise

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract depth ROI
            roi_depth = depth_image[y:y + h, x:x + w].astype(float)
            roi_depth = roi_depth[roi_depth > 0]

            if roi_depth.size > 0:
                median_depth = np.median(roi_depth) * depth_scale

                # Convert pixel width/height to meters
                width_m = (w * median_depth) / fx
                height_m = (h * median_depth) / fy
                depth_m = median_depth  # toward camera

                dimensions_text = f"L: {width_m:.2f}m, B: {height_m:.2f}m, H: {depth_m:.2f}m"
                cv2.putText(color_image, dimensions_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        combined = np.hstack((color_image, depth_colormap))

        cv2.imshow("RealSense Live Object Measurement", combined)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
