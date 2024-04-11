import os
import csv
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo

def calculate_rot_matrix(roll, pitch, yaw):
    # Calculate rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def calculate_transformation_matrix(x, y, z, roll, pitch, yaw):
    R = calculate_rot_matrix(roll, pitch, yaw)
    T = np.array([[R[0, 0], R[0, 1], R[0, 2], x],
                  [R[1, 0], R[1, 1], R[1, 2], y],
                  [R[2, 0], R[2, 1], R[2, 2], z],
                  [0,        0,        0,     1]])
    return T


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        # Define a custom QoS profile with the correct access to reliability and durability
        custom_qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        # Use the custom QoS profile for the subscription
        self.subscription = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            custom_qos_profile)
        self.pose_subscription = self.create_subscription(
            Pose,
            '/drone/gt_pose',
            self.gt_pose_callback,
            custom_qos_profile)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/drone/front/camera_info',
            self.camera_info_callback,
            custom_qos_profile)
        self.bridge = CvBridge()
        self.objects_data = {
            "asphalt_plane": {
                "pose": [0.027118, -0.028406, 0],
                "orientation": [0, 0, 0],
                "size": [20.000000, 20.000000, 0.100000],
            },
            "control_console": {
                "pose": [0.044032, 9.471480, 0],
                "orientation": [0, 0, 0],
                "size": [1.780000, 0.050000, 2.602250],
            },

            "unit_box": {
                "pose": [9.406795, -9.399209, 0.550000],
                "orientation": [0, 0, 0.000034],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone": {
                "pose": [9.424715, 9.274771, 0.550000],
                "orientation": [0, 0, 0.000040],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone": {
                "pose": [-9.261930, 9.274990, 0.550000],
                "orientation": [0, 0, 0.000016],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone_clone": {
                "pose": [-8.978432, -9.372633, 0.549995],
                "orientation": [-0.000010, 0.000001, 0.111119],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number1": {
                "pose": [-3.443590, -9.539260, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number2": {
                "pose": [-9.348060, 5.928570, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number3": {
                "pose": [4.345270, 9.360890, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number4": {
                "pose": [9.514570, -4.511090, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "Construction Cone": {
                "pose": [3.440740, 0.667633, 0.049999],
                "orientation": [-0.000002, 0.000002, -0.004192],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_0": {
                "pose": [2.936450, 0.915504, 0.050000],
                "orientation": [0, 0, -0.001118],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_1": {
                "pose": [2.419270, 1.061460, 0.049991],
                "orientation": [0, -0.000002, 0.000614],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_2": {
                "pose": [1.916210, 1.225750, 0.050000],
                "orientation": [0, 0, -0.000372],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_3": {
                "pose": [1.412490, 1.358500, 0.049991],
                "orientation": [0, 0, -0.014740],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_4": {
                "pose": [0.906406, 1.475050, 0.050000],
                "orientation": [0, 0, 0.002556],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_5": {
                "pose": [0.377529, 1.550320, 0.050001],
                "orientation": [0, -0.000002, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_6": {
                "pose": [-0.141692, 1.602430, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_7": {
                "pose": [-0.669365, 1.771780, 0.049990],
                "orientation": [-0.000002, 0, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_8": {
                "pose": [-1.185310, 1.961130, 0.050000],
                "orientation": [0.000002, -0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_9": {
                "pose": [-1.720400, 2.194210, 0.049992],
                "orientation": [0, 0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_10": {
                "pose": [-2.251430, 2.447520, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Dumpster": {
                "pose": [1.105810, -9.136140, 0.051370],
                "orientation": [0.000010, 0, -3.136771],
                "size": [1.773333, 0.886666, 0.686666],
            },
            "hoop_red": {
                "pose": [-3.397180, -10.136400, 0],
                "orientation": [0, 0, 0],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "hoop_red_0": {
                "pose": [-9.971850, 3.162810, 0],
                "orientation": [0, 0, -1.556320],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "person_standing": {
                "pose": [0.739916, -7.874418, 0.050000],
                "orientation": [0, -0.000002, 0.001970],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_walking": {
                "pose": [-7.531469, 5.773241, 0.049999],
                "orientation": [0, 0.000006, 0.005452],
                "size": [0.350000, 0.750000, 0.020000],
            },
            "person_standing_0": {
                "pose": [0.592537, 8.466297, 0.050000],
                "orientation": [0, -0.000002, -3.140959],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_1": {
                "pose": [-0.195836, 2.403852, 0.050000],
                "orientation": [0, 0.000002, 0.001718],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_2": {
                "pose": [-9.561253, -8.743303, 0.294097],
                "orientation": [-1.636140, 0.909863, 0.015181],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_3": {
                "pose": [1.827380, 6.648900, 0.000537],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_4": {
                "pose": [-2.140890, 5.534660, 0.002202],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_5": {
                "pose": [-2.494090, 3.011030, -0.007182],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_6": {
                "pose": [4.132250, 5.527540, 0.007169],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_7": {
                "pose": [-3.036670, -5.568780, -0.009657],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
        }

        # Camera parameters
        self.horizontal_fov = 2.09  # Horizontal field of view in radians
        self.image_width = 640  # Image width
        self.image_height = 360  # Image height
        self.focal_length_x = self.image_width / (2 * np.tan(self.horizontal_fov / 2))
        self.focal_length_y = self.focal_length_x  # Assuming square pixels for simplicity
        self.c_x = self.image_width / 2
        self.c_y = self.image_height / 2

        # Camera intrinsic matrix (projection matrix) updated
        self.K = np.array([
            [self.focal_length_x, 0, self.c_x],
            [0, self.focal_length_y, self.c_y],
            [0, 0, 1]
        ])

        # Check that the optical center is at the middle of the width and height of the image
        assert self.c_x == self.image_width / 2, "Optical center X is not in the middle of the image width!"
        assert self.c_y == self.image_height / 2, "The optical center Y is not at the middle of the image height!"

        # Check that horizontal_fov is set correctly
        assert self.horizontal_fov == 2.09, "Horizontal FOV does not correspond to the value in the URDF!"

        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        # Adding a new key for each object to store the transformation matrix
        for obj_name, obj_data in self.objects_data.items():
            pose = obj_data['pose']
            orientation = obj_data['orientation']
            # Calculate the rotation matrix using the calculate_rot_matrix function
            R_obj = calculate_rot_matrix(*orientation)
            # Create the transformation matrix for each object
            self.objects_data[obj_name]['transformation_matrix'] = calculate_transformation_matrix(*pose, *orientation)

    camera_transformation = np.eye(4)  # Initialize as a 4x4 identity matrix
    camera_transformation[:3, 3] = [0.2, 0, 0]  # Apply the translation of 0.2 meters on the X axis

    def camera_info_callback(self, msg):
        # Get the current time for timestamp
        current_time = self.get_clock().now().to_msg()
        self.K = np.array(msg.k).reshape((3, 3))
        # Print intrinsic parameters
        print(f'Intrinsic Matrix (K):\n{self.K}')
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            processed_image = self.project_and_draw_objects(cv_image)
            cv2.imshow("Imagine cu obiecte", processed_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            pass
            # self.get_logger().error(f"Could not convert ROS Image message to OpenCV image: {str(e)}")

    def gt_pose_callback(self, msg):
        # Extract position
        position = np.array([msg.position.x, msg.position.y, msg.position.z])

        # Extract orientation and convert to rotation matrix
        orientation_q = [msg.orientation.x, msg.orientation.y, msg.orientation.z,
                         msg.orientation.w]
        self.R = R.from_quat(orientation_q).as_matrix()

        # Update extrinsic parameters
        self.t = position.reshape((3, 1))

        print(f'Position: {self.t.transpose()}')
        print(f'Rotation Matrix:\n{self.R}')

    def project_and_draw_objects(self, image):

        # Logic for transformation chain:
        # retrieve the information about the obstacle
        # pos,rot = ?

        # the transformation matrix from obstacle to world
        # transform_obstacol_world=?

        # we are retrieving information about the drone
        # pos,rot = ?

        # the drone-to-world transformation matrix
        # transform_drone_world = ?

        # transformation matrix from world to drone (inverse)
        # transform_world_drone = ?

        # the transformation matrix from the drone to the camera
        # transform_drone_camera = ?

        # the projection matrix
        # projection_matrix = ?

        # we choose a point in 3D
        # x,y,z = ?

        # prepare for transformation
        # point_matrix_3D = ?

        # apply transformation
        # point_matrix_2D = point_matrix_3D @ transform_obstacol_world @ transform_world_drone @transform_drone_camera @ projection_matrix

        # Get the position and orientation of the obstacle (cone)
        obstacle_pos = np.array(self.objects_data["Construction Cone"]["pose"])
        obstacle_rot = np.array(self.objects_data["Construction Cone"]["orientation"])

        # Compute the obstacle-to-world transformation matrix
        transform_obstacle_world = calculate_transformation_matrix(*obstacle_pos, *obstacle_rot)

        # Get the position and orientation of the drone
        drone_pos = np.array([self.t[0, 0], self.t[1, 0], self.t[2, 0]])
        drone_rot = R.from_matrix(self.R).as_euler('xyz')

        # Compute the drone-to-world transformation matrix
        transform_drone_world = calculate_transformation_matrix(*drone_pos, *drone_rot)

        # Invert the transform to get the matrix from world to drone
        transform_world_drone = np.linalg.inv(transform_drone_world)

        # Defines the camera offset in drone coordinates
        camera_offset_drone_frame = np.array([0.2, 0, 0])

        # Defines the rotation required to align the camera with the Z axis of the world
        rotation_matrix_camera = R.from_euler('y', np.pi / 2).as_matrix()

        # Combine translation and rotation to get the full transformation from drone to camera
        transform_drone_camera = np.eye(4)
        transform_drone_camera[:3, :3] = rotation_matrix_camera
        transform_drone_camera[:3, 3] = camera_offset_drone_frame

        # Project the 3D point in room space
        # Here, adjust the X,Y,Z values to move the point in the image (for debug only)
        x, y, z = obstacle_pos + np.array([-0.2, 0, 0])
        point_matrix_3D = np.array([x, y, z, 1])

        point_in_world_space = transform_obstacle_world.dot(point_matrix_3D)

        print(f'3D Point in Obstacle Space: {point_matrix_3D}')
        print(f'3D Point in World Space: {point_in_world_space}')
        point_in_camera_space = transform_world_drone.dot(transform_drone_camera).dot(point_in_world_space)
        print(f'3D Point in Camera Space: {point_in_camera_space}')

        # Project the point into the 2D image space using the camera projection matrix
        point_matrix_2D_homogeneous = self.K.dot(point_in_camera_space[:3])

        # Normalize the 2D coordinates to get the pixel position
        if point_matrix_2D_homogeneous[2] != 0:
            point_2D = point_matrix_2D_homogeneous[:2] / point_matrix_2D_homogeneous[2]
        else:
            # If Z is zero, we can't do the division; we return the image unchanged
            print("Division by zero in 2D point projection.")
            return image

        # Check if the point is within the image bounds and draw it
        if 0 <= point_2D[0] < self.image_width and 0 <= point_2D[1] < self.image_height:
            cv2.circle(image, (int(point_2D[0]), int(point_2D[1])), radius=2, color=(0, 255, 0), thickness=-1)
        else:
            print("The point is outside the bounds of the image.")

        return image


# ROS2 Node execution
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    cv2.namedWindow("Image with objects", cv2.WINDOW_NORMAL)  # Initialize the window
    rclpy.spin(image_subscriber)
    cv2.destroyAllWindows()  # Destroy the window when done
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
