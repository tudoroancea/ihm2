from matplotlib import scale
import numpy as np
import rclpy
# import track_database as tdb
from diagnostic_msgs.msg import *
from geometry_msgs.msg import *
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import *
# from rclpy.subscription import Subscription
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, Header
# from tf2_geometry_msgs import from_msg_msg
from scipy.spatial.transform import *
from visualization_msgs.msg import Marker, MarkerArray

def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    t0 = np.cos(yaw * 0.5)
    t1 = np.sin(yaw * 0.5)
    t2 = np.cos(roll * 0.5)
    t3 = np.sin(roll * 0.5)
    t4 = np.cos(pitch * 0.5)
    t5 = np.sin(pitch * 0.5)

    return [
        t0 * t2 * t4 + t1 * t3 * t5,  # w
        t0 * t3 * t4 - t1 * t2 * t5,  # x
        t0 * t2 * t5 + t1 * t3 * t4,  # y
        t1 * t2 * t4 - t0 * t3 * t5,  # z
    ]
marker_colors = {
    "red": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    "green": ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    "blue": ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    "yellow": ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    "orange": ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    "purple": ColorRGBA(r=0.5, g=0.0, b=0.5, a=1.0),
    "magenta": ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
    "cyan": ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
    "light_blue": ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0),
    "dark_blue": ColorRGBA(r=0.0, g=0.0, b=0.5, a=1.0),
    "brown": ColorRGBA(r=0.5, g=0.25, b=0.0, a=1.0),
    "black": ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
    "white": ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
    "gray": ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
    "light_gray": ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0),
    "dark_gray": ColorRGBA(r=0.25, g=0.25, b=0.25, a=1.0),
}

class MeshPublisherNode(Node):
    def __init__(self):
        super().__init__('mesh_publisher')
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.publisher = self.create_publisher(MarkerArray, 'mesh', 10)
    
    def timer_callback(self):
        self.get_logger().info('publishing mesh')
        cone_marker = lambda id, X, Y, color : Marker(
                header=Header(frame_id="world"),
                ns="cones",
                id=id,
                type=Marker.ARROW,
                action=Marker.MODIFY,
                scale=Vector3(x=0.0, y=0.228, z=0.325),
                color=marker_colors[color],
                points=[
                    Point(
                        x=X,
                        y=Y,
                        z=0.0,
                    ),
                    Point(
                        x=X,
                        y=Y,
                        z=0.325,
                    ),
                ],
            )
        qw, qx, qy, qz = rpy_to_quaternion(0, 0, np.pi)        
        q = Quaternion(w=qw, x=qx, y=qy, z=qz)
        self.publisher.publish(
            MarkerArray(
                markers=[
                    cone_marker(0, 0.0, 0.0, "blue"),
                    cone_marker(1, 1.0, 0.0, "yellow"),
                    Marker(
                        header=Header(frame_id="world"),
                        ns="mesh",
                        type=Marker.MESH_RESOURCE,
                        action=Marker.MODIFY,
                        mesh_resource="https://github.com/tudoroancea/ihm2/releases/download/lego-lrt4/lego-lrt4.stl",
                        scale=Vector3(x=0.03, y=0.03, z=0.03),
                        color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
                        pose=Pose(
                            position=Point(
                                x=2.0,
                                y=0.0,
                                z=0.0,
                            ),
                            orientation=q,
                        ),
                    ),
                ]
            )
        )

def main():
    rclpy.init()
    node = MeshPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
