import pyrealsense2 as rs # imported module pyrelease
import numpy as np # imported numpy module
import cv2 # imported cv2 module

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create an align object to align depth and color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Get frame size
frame_width = 640
frame_height = 480

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Align depth frame to color frame (optional)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize the depth image to 8-bit for contour detection
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colorization to depth frame for visualization (optional)
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Threshold or detect the object in the depth frame
        _, thresholded_depth = cv2.threshold(depth_image_normalized, 100, 255, cv2.THRESH_BINARY)

        # Find contours of the object in the thresholded image
        contours, _ = cv2.findContours(thresholded_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get the bounding box of the object
                x, y, w, h = cv2.boundingRect(contour)

                # Ensure that the coordinates are within valid bounds
                x = min(x, frame_width - 1)
                y = min(y, frame_height - 1)
                x2 = min(x + w, frame_width - 1)
                y2 = min(y + h, frame_height - 1)

                # Measure the object in 3D space using depth data (center of bounding box)
                object_depth_top_left = depth_frame.get_distance(x, y)
                object_depth_bottom_right = depth_frame.get_distance(x2, y2)

                # Calculate the real-world length and breadth (in meters)
                length = (object_depth_bottom_right - object_depth_top_left) * w
                breadth = (object_depth_bottom_right - object_depth_top_left) * h

                # Get the object's height (you can take the average of the depths)
                avg_depth = np.mean(depth_image[y:y+h, x:x+w])  # Average depth of the bounding box area
                height = avg_depth  # Height corresponds to depth

                # Print or display the object length, breadth, and height
                print(f"Object Length: {length:.2f} meters")
                print(f"Object Breadth: {breadth:.2f} meters")
                print(f"Object Height (Depth): {height:.2f} meters")

                # Draw the bounding box and show the image
                cv2.rectangle(depth_colormap, (x, y), (x2, y2), (0, 255, 0), 2)

        # Show the depth image
        cv2.imshow("Depth Image", depth_colormap)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
