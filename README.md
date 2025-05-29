# cloudpoint2camera
A script for projecting multiple point cloud pcd files onto the camera plane.

Extracting point clouds using the pcl_ros package/tool:

rosrun pcl_ros bag_to_pcd \
  /path-to-data/data.bag \
  /lidar-topic \
  /path-to-result/cloud-points

Extracting pictures using extract_frame.py.

It requires camera intrinsic parameters and the transformation relationship with the lidar. It can match and project multiple point clouds and images in a folder at one time to generate a depth map.
