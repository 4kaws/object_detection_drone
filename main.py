
# !/usr/bin/env python3
import os
import csv
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor  # Import SingleThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(Image, '/drone/front/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.last_save_time = time.time()
        self.save_interval = 5  # seconds
        self.image_folder = '/home/andrei/ros2_ws/images'
        self.csv_file_path = '/home/andrei/ros2_ws/dataset.csv'
        os.makedirs(self.image_folder, exist_ok=True)
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow \
            (["image_name", "object_name", "x", "y", "z", "rx", "ry", "rz", "w", "l", "h", "x_min_bb", "y_min_bb", "x_max_bb", "y_max_bb"])
        self.image_counter = 0
        self.objects_data = {
            "Oak_Tree2_clone_0": {"pose": (4.949010, -2.283450, 0.000786), "size": (0.5, 0.35, 0.02), "scale": (1.0, 1.0, 1.0)},
            "Oak_Tree2_clone_1": {"pose": (-5.659300, -3.974970, -0.005122), "size": (0.5, 0.35, 0.02), "scale": (1.0, 1.0, 1.0)},
            "Oak_Tree2_clone_2": {"pose": (1.541940, 3.084460, -0.005634), "size": (0.5, 0.35, 0.02), "scale": (1.0, 1.0, 1.0)},
            "Oak_Tree2_clone_3": {"pose": (-6.674920, 1.072040, 0.009168), "size": (0.5, 0.35, 0.02), "scale": (1.0, 1.0, 1.0)},
            "Oak_Tree2_clone_4": {"pose": (-8.605900, -3.887810, 0.009493), "size": (0.5, 0.35, 0.02), "scale": (1.0, 1.0, 1.0)},
            "unit_box": {"pose": (9.406795, -9.399209, 0.550000), "size": (1.0, 1.0, 1.0), "scale": (1.0, 1.0, 1.0)},
            "unit_box_clone": {"pose": (9.424715, 9.274771, 0.550000), "size": (1.0, 1.0, 1.0), "scale": (1.0, 1.0, 1.0)},
            "unit_box_clone_clone": {"pose": (-9.261930, 9.274990, 0.550000), "size": (1.0, 1.0, 1.0), "scale": (1.0, 1.0, 1.0)},
            "unit_box_clone_clone_clone": {"pose": (-8.978432, -9.372638, 0.550000), "size": (1.0, 1.0, 1.0), "scale": (1.0, 1.0, 1.0)},
            "number1": {"pose": (-3.443590, -9.539260, 0.400000), "size": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
            "number2": {"pose": (-9.348060, 5.928570, 0.400000), "size": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
            "number3": {"pose": (4.345270, 9.360890, 0.400000), "size": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
            "number4": {"pose": (9.514570, -4.511090, 0.400000), "size": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
            "control_console": {"pose": (0.044032, 9.471480, 0.000000), "size": (1.78, 0.05, 2.602250), "scale": (1.0, 1.0, 1.0)},
            "Construction Cone": {"pose": (3.440740, 0.667633, 0.049999), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_1": {"pose": (2.419270, 1.061460, 0.049991), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_2": {"pose": (1.916210, 1.225750, 0.050000), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_3": {"pose": (1.412490, 1.358500, 0.049991), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_4": {"pose": (0.906406, 1.475050, 0.050000), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_5": {"pose": (0.377529, 1.550320, 0.050001), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_6": {"pose": (-0.141692, 1.602430, 0.050000), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_7": {"pose": (-0.669365, 1.771780, 0.049990), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_8": {"pose": (-1.185310, 1.961130, 0.050000), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_9": {"pose": (-1.720400, 2.194210, 0.049992), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Construction Cone_10": {"pose": (-2.251430, 2.447520, 0.050000), "size": (0.5, 0.5, 1.0), "scale": (10.0, 10.0, 10.0)},
            "Dumpster": {"pose": (1.105810, -9.136140, 0.051370), "size": (0.5, 0.5, 1.0), "scale": (1.5, 1.5, 1.5)},
            "hoop_red": {"pose": (-3.397180, -10.136400, 0.000000), "size": (0.3, 0.1, 5.0), "scale": (1.5, 1.5, 1.5)},
            "hoop_red_0": {"pose": (-9.971850, 3.162810, 0.000000), "size": (0.3, 0.1, 5.0), "scale": (1.5, 1.5, 1.5)},
            "person_standing": {"pose": (0.739916, -7.874418, 0.050000), "size": (0.5, 0.35, 0.2), "scale": (1.0, 1.0, 1.0)},
            "person_standing_0": {"pose": (0.592537, 8.466297, 0.050000), "size": (0.5, 0.35, 0.2), "scale": (1.0, 1.0, 1.0)},
            "person_standing_1": {"pose": (-0.195836, 2.403852, 0.050000), "size": (0.5, 0.35, 0.2), "scale": (1.0, 1.0, 1.0)},
            "person_standing_2": {"pose": (-9.561234, -8.743356, 0.294077), "size": (0.5, 0.35, 0.2), "scale": (1.0, 1.0, 1.0)},
            "person_walking": {"pose": (-7.531469, 5.773241, 0.049999), "size": (0.35, 0.75, 0.2), "scale": (1.0, 1.0, 1.0)},
        }

        # Camera specifications
        fov_horizontal = 2.09  # Horizontal field of view in radians
        image_width = 640      # Image width in pixels
        image_height = 360     # Image height in pixels
        focal_length_x = image_width / (2 * np.tan(fov_horizontal / 2))
        focal_length_y = focal_length_x
        c_x = image_width / 2
        c_y = image_height / 2
        self.K = np.array([[focal_length_x, 0, c_x], [0, focal_length_y, c_y], [0, 0, 1]])  # Intrinsic matrix
        self.R = np.eye(3)  # Assuming no rotation at start
        self.t = np.array([[0, 0, 0]]).T  # Translation vector is zero at start
        self.camera_projection_matrix = None  # Camera projection matrix

    def calc(self):
        # get camera pos and orientation
        self.R = np.eye(3)
        self.t = np.array([[0, 0, 0]]).T
        self.camera_projection_matrix = np.dot(self.K, np.hstack((self.R, self.t)))

    def project_to_image(self, world_point):
        self.calc()
        world_point_homogeneous = np.append(world_point, 1)
        image_point_homogeneous = np.dot(self.camera_projection_matrix, world_point_homogeneous)
        u = image_point_homogeneous[0] / image_point_homogeneous[2]
        v = image_point_homogeneous[1] / image_point_homogeneous[2]
        return int(u), int(v)

    def image_callback(self, msg):
        print("Processing image...")
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # Calculate bounding boxes for each object
                bounding_boxes = []
                for object_id, data in self.objects_data.items():
                    pose = np.array(data["pose"])
                    size = np.array(data["size"]) * np.array(data["scale"])
                    bounding_box = self.calculate_bounding_box(pose, size)
                    if bounding_box:  # If a bounding box was calculated
                        bounding_boxes.append((object_id, bounding_box))

                # Save image
                image_name = f'image_{self.image_counter}.png'
                cv2.imwrite(os.path.join(self.image_folder, image_name), cv_image)
                self.save_object_data(image_name, bounding_boxes)
                self.image_counter += 1
                self.last_save_time = current_time
            except CvBridgeError as e:
                self.get_logger().error('CVBridgeError: %s' % str(e))

    def calculate_bounding_box(self, pose, size):
        half_size = size / 2
        corners = [pose + half_size * np.array([sx, sy, sz]) for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]]
        projected_corners = [self.project_to_image(corner) for corner in corners]
        if not projected_corners:
            return None  # Skip if projection fails
        x_coords, y_coords = zip(*projected_corners)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return x_min, y_min, x_max, y_max

    def save_object_data(self, image_name, bounding_boxes):
        for object_id, (x_min, y_min, x_max, y_max) in bounding_boxes:
            data = self.objects_data[object_id]
            pose = data['pose']
            size = data['size']
            scale = data['scale']
            self.csv_writer.writerow([
                image_name,
                object_id,
                pose[0], pose[1], pose[2],  # x, y, z
                size[0], size[1], size[2],  # width, length, height
                scale[0], scale[1], scale[2],  # scale x, y, z
                x_min, y_min, x_max, y_max  # Bounding box coordinates
            ])


    def __del__(self):
        self.csv_file.close()

def main(args=None):
    print('GOGO')
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    executor = SingleThreadedExecutor()

    try:
        executor.add_node(image_subscriber)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Safe shutdown process
        if rclpy.ok():
            image_subscriber.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

