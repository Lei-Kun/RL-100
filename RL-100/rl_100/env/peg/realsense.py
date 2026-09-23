import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import fpsample


def depth2pc(depth, camera_intrinsics, camera_pose=np.eye(4)):
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    z = depth.flatten()

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=1)[z > 0]

    points_camera_h = np.concatenate((points_camera, np.ones((points_camera.shape[0], 1))), axis=1)  # Nx4
    points_world = (camera_pose @ points_camera_h.T).T[:, :3]

    return points_world


def point_cloud_downsample(point_cloud, num_points):
    # bounding box filter
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    bounding_box_mask = (x > 0.2) & (x < 1) & (y > -0.2) & (y < 0.2) & (z > 0.03) & (z < 0.3)
    point_cloud = point_cloud[bounding_box_mask]


    # FPS sampling
    if len(point_cloud) < num_points:
        point_cloud = np.concatenate([point_cloud] * (num_points // len(point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(point_cloud, num_points)
    point_cloud = point_cloud[sample_idx]

    return point_cloud


camera_intrinsics = (1346.89, 1346.89, 962.28, 559.75)
# modify after calibrating extrinsics 
X_root_camera = np.array([
    [-0.28586096,  0.63197252, -0.72034314,  0.7757766 ],
    [ 0.95784787,  0.16610085, -0.23438848,  0.13477495],
    [-0.02847747, -0.75698166, -0.65281528,  0.493779  ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
# rot_mat = X_root_camera[:3, :3]
# rot_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
# print(rot_euler)
# rot_euler[0] -= 0.2
# rot_euler[1] += 0.5
# rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
# print(rot_mat)
# X_root_camera[:3, :3] = rot_mat


class RealSense(object):
    def __init__(
        self,
        fps=30,
        enable_color=False,
        color_width=640,
        color_height=480,
        depth_width=320,
        depth_height=240,
        num_points=2048,
        device_serial=None,
        enable_depth=True,
        color_fps=None,
        depth_fps=None,
    ):
        import pyrealsense2 as rs

        self.enable_color = bool(enable_color)
        self.enable_depth = bool(enable_depth)
        if not self.enable_color and not self.enable_depth:
            raise ValueError('At least one camera stream must be enabled')
        color_fps = fps if color_fps is None else color_fps
        depth_fps = fps if depth_fps is None else depth_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if device_serial is not None:
            self.config.enable_device(device_serial)
        if self.enable_color:
            self.config.enable_stream(
                rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps
            )
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps
            )
        self.align = rs.align(rs.stream.color) if self.enable_color else None

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        frames = self.pipeline.wait_for_frames()
        if self.enable_depth:
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        if require_pc and not self.enable_depth:
            raise ValueError('Point cloud requires the depth stream')
        while True:
            frames = self.pipeline.wait_for_frames()

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            color_frame = frames.get_color_frame() if self.enable_color else None
            depth_frame = frames.get_depth_frame() if self.enable_depth else None

            if (not self.enable_color or color_frame) and (not self.enable_depth or depth_frame):
                break

        color_image = np.array(color_frame.get_data()) if color_frame else None
        depth_image = np.array(depth_frame.get_data()) if depth_frame else None

        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            'color': color_image,
            'depth': depth_image,
            'depth_scale': self.depth_scale if self.enable_depth else None,
            'point_cloud': point_cloud
        }


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    while True:
        frame = camera.get_frame(require_pc=True)
        # print('timestamp:', frame['timestamp'])

        server.scene.add_frame(
            f'camera_pose',
            wxyz=R.from_matrix(X_root_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=X_root_camera[:3, 3],
            axes_length=0.2,
            axes_radius=0.006
        )

        server.scene.add_point_cloud(
            'pc',
            frame['point_cloud'],
            point_size=0.002,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        time.sleep(0.033)
