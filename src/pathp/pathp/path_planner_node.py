#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

import pathp.fsd_path_planning.full_pipeline.full_pipeline as pp


from enum import IntEnum

class ConeTypes(IntEnum):
    """
    Enum for all possible cone types
    """

    UNKNOWN = 0
    RIGHT = YELLOW = 1
    LEFT = BLUE = 2
    START_FINISH_AREA = ORANGE_SMALL = 3
    START_FINISH_LINE = ORANGE_BIG = 4


class MissionTypes(IntEnum):
    """
    Enum for each mission type
    """

    (
        none,
        acceleration,
        skidpad,
        autocross,
        trackdrive,
        ebs_test,
        inspection,
        manual_driving,
    ) = range(8)

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # Parameters
        self.declare_parameter('mission_type', 'trackdrive') # trackdrive, skidpad, acceleration
        self.declare_parameter('cones_topic', '/ground_truth/cones')
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('path_topic', '/path_planner/path')
        self.declare_parameter('frame_id', 'map')

        mission_str = self.get_parameter('mission_type').value
        self.cones_topic = self.get_parameter('cones_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        # Initialize PathPlanner
        if mission_str == 'skidpad':
            self.mission = MissionTypes.skidpad
        elif mission_str == 'acceleration':
            self.mission = MissionTypes.acceleration
        else:
            self.mission = MissionTypes.trackdrive
        
        self.planner = pp.PathPlanner(self.mission)

        # QoS
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)

        # Subscribers
        self.create_subscription(ConeArrayWithCovariance, self.cones_topic, self.cone_callback, qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)

        # Publishers
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)

        # State
        self.vehicle_position = np.zeros(2)
        self.vehicle_direction = np.array([1.0, 0.0])
        self.cones = [np.zeros((0, 2)) for _ in ConeTypes]
        self.got_odom = False

        self.get_logger().info(f"PathPlannerNode started. Mission: {mission_str}")

    def odom_callback(self, msg: Odometry):
        self.vehicle_position[0] = msg.pose.pose.position.x
        self.vehicle_position[1] = msg.pose.pose.position.y
        
        # Quaternion to direction vector
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.vehicle_direction[0] = np.cos(yaw)
        self.vehicle_direction[1] = np.sin(yaw)
        
        self.got_odom = True

    def cone_callback(self, msg: ConeArrayWithCovariance):
        if not self.got_odom:
            return

        # Extract cones
        def extract_cones(cone_list):
            return np.array([[c.point.x, c.point.y] for c in cone_list]) if cone_list else np.zeros((0, 2))

        self.cones[ConeTypes.BLUE] = extract_cones(msg.blue_cones)
        self.cones[ConeTypes.YELLOW] = extract_cones(msg.yellow_cones)
        self.cones[ConeTypes.ORANGE_SMALL] = extract_cones(msg.orange_cones)
        self.cones[ConeTypes.ORANGE_BIG] = extract_cones(msg.big_orange_cones)

        # Calculate Path
        try:
            path = self.planner.calculate_path_in_global_frame(
                self.cones,
                self.vehicle_position,
                self.vehicle_direction
            )
            
            self.publish_path(path)
            
        except Exception as e:
            self.get_logger().error(f"Path calculation failed: {e}")

    def publish_path(self, path_points):
        # path_points is Nx4 (s, x, y, k)
        if path_points is None or len(path_points) == 0:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.frame_id

        for pt in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[1]
            pose.pose.position.y = pt[2]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
