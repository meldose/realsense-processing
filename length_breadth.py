
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

