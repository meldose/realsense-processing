import pyrealsense2 as rs
import numpy as np
import cv2
import os

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

cv2.imwrite("color_image.jpg", color_image)
cv2.imwrite("depth_image.png", depth_image)

print("Saved color_image.jpg and depth_image.png")

pipeline.stop()
