import pyrealsense2 as rs
import numpy as np
import cv2

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize the depth image to 8-bit for contour detection
        # Normalize the depth image to the range [0, 255]
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colorization to depth frame for visualization (optional)
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Threshold or detect the object in the depth frame
        # For simplicity, we're using a simple thresholding approach here.
        _, thresholded_depth = cv2.threshold(depth_image_normalized, 100, 255, cv2.THRESH_BINARY)

        # Find contours of the object in the thresholded image
        contours, _ = cv2.findContours(thresholded_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get the bounding box of the object
                x, y, w, h = cv2.boundingRect(contour)

                # Measure the object in 3D space using depth data
                object_depth = depth_frame.get_distance(x + w // 2, y + h // 2)

                # Print or display the object length (width) and breadth (height)
                print(f"Object Depth: {object_depth:.2f} meters")
                print(f"Object Length: {w} pixels")
                print(f"Object Breadth: {h} pixels")

                # Draw the bounding box and show the image
                cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the depth image
        cv2.imshow("Depth Image", depth_colormap)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
