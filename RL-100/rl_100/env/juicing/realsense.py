import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector
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
    # bounding_box_mask = ((z > 0.03) & (z < 0.15) & (x > 0.32) & (x < 0.65) & (y > -0.29) & (y < -0.05))    

    # below: pre box
    # bounding_box_mask = ((z > 0.03) & (z < 0.15) & (x > 0.35) & (x < 0.626) & (y > -0.267) & (y < -0.079)) | \
    # below: large box
    # bounding_box_mask = ((z > 0.03) & (z < 0.15) & (x > 0.32) & (x < 0.66) & (y > -0.295) & (y < -0.05)) | \

    bounding_box_mask = ((z > 0.03) & (z < 0.15) & (x > 0.35) & (x < 0.626) & (y > -0.267) & (y < -0.079)) | \
                         ((z > 0.22) & (z < 0.29) & (x > 0.38) & (x < 0.51) & (y > 0.13) & (y < 0.27)) | \
                         ((z > 0.23) & (z < 0.34) & (x > 0.315) & (x < 0.475) & (y > 0.06) & (y < 0.2))
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


class RealSense(object):
    def __init__(
            self,
            color_width=1920,
            color_height=1080,
            fps=30,
            enable_depth=True,
            depth_width=320,
            depth_height=240,
            apriltag_families="tagStandard41h12",
            num_points=10240
        ):
        self.enable_depth = enable_depth
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

        self.detector = Detector(families=apriltag_families)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        self.intrinsics = camera_intrinsics
        frames = self.pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        # self.color_intrinsics = (color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
        if self.enable_depth:
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
            print(f"Depth Intrinsics: {self.depth_intrinsics}")

    def stop(self):
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            # frames = self.align.process(frames)

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            # color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if depth_frame:
                break

        # color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            # 'color': color_image,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud
        }
    
    def detect_apriltag(self, color, tag_size=0.03 * 5 / 9, tag_num=3):
        detections = self.detector.detect(
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=self.intrinsics,
            tag_size=tag_size
        )
        # print(f'{len(detections)} tags detected.')

        tag_poses = [None] * tag_num
        for detection in detections:
            if detection.tag_id >= tag_num:
                continue
            X_camera_tag = np.eye(4)
            X_camera_tag[:3, :3] = detection.pose_R
            X_camera_tag[:3, 3] = detection.pose_t.flatten()
            X_root_tag = X_root_camera @ X_camera_tag
            tag_poses[detection.tag_id] = X_root_tag
        
        return tag_poses


def add_world_grid(server):
    """添加简洁明了的世界坐标系刻度"""
    
    # 添加增强的世界坐标轴
    server.scene.add_frame(
        'world_origin',
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
        axes_length=0.4,
        axes_radius=0.01
    )
    
    # 创建主要的刻度线
    def create_scale_lines(axis_name, positions, color, direction):
        """创建刻度线"""
        lines = []
        for pos in positions:
            if direction == 'x':
                # X轴刻度：垂直短线
                lines.extend([
                    [pos, -0.05, 0], [pos, 0.05, 0],    # Y方向短线
                    [pos, 0, -0.05], [pos, 0, 0.05]     # Z方向短线
                ])
            elif direction == 'y':
                # Y轴刻度：水平短线  
                lines.extend([
                    [-0.05, pos, 0], [0.05, pos, 0],    # X方向短线
                    [0, pos, -0.05], [0, pos, 0.05]     # Z方向短线
                ])
            elif direction == 'z':
                # Z轴刻度：水平短线
                lines.extend([
                    [-0.05, 0, pos], [0.05, 0, pos],    # X方向短线
                    [0, -0.05, pos], [0, 0.05, pos]     # Y方向短线
                ])
        
        if lines:
            server.scene.add_point_cloud(
                f'{axis_name}_scale_lines',
                np.array(lines),
                point_size=0.004,
                point_shape="circle",
                colors=color
            )
    
    # X轴刻度（红色）：每0.1单位一个刻度
    x_positions = np.arange(0.0, 1.0 + 0.1, 0.1)
    create_scale_lines('x', x_positions, (255, 0, 0), 'x')
    
    # Y轴刻度（绿色）：每0.1单位一个刻度，覆盖负数和正数范围
    y_positions = np.arange(-1.0, 0.5 + 0.1, 0.1)
    create_scale_lines('y', y_positions, (0, 255, 0), 'y')
    
    # Z轴刻度（蓝色）：每0.05单位一个刻度
    z_positions = np.arange(0.0, 0.6 + 0.05, 0.05)
    create_scale_lines('z', z_positions, (0, 0, 255), 'z')
    
    # 添加关键参考平面
    plane_points = []
    
    # Z=0 平面（地面，浅灰色网格）
    for x in np.arange(0.0, 1.0, 0.1):
        for y in np.arange(-1.0, 0.5, 0.1):
            plane_points.append([x, y, 0.0])
    
    if plane_points:
        server.scene.add_point_cloud(
            'ground_plane',
            np.array(plane_points),
            point_size=0.001,
            point_shape="circle",
            colors=(200, 200, 200)  # 浅灰色
        )
    
    # 添加关键高度平面标记
    key_z_levels = [0.03, 0.055, 0.2, 0.5]  # 你bounding box中的关键Z值
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 255)]
    
    for i, z in enumerate(key_z_levels):
        key_points = []
        # 在关键区域画几个标记点
        for x in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for y in [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]:
                key_points.append([x, y, z])
        
        server.scene.add_point_cloud(
            f'z_level_{z}',
            np.array(key_points),
            point_size=0.002,
            point_shape="circle",
            colors=colors[i]
        )
    
    # 添加bounding box的可视化边框
    bbox_corners = []
    # 第一个bounding box: (z > 0.0) & (z < 0.5) & (x > 0.33) & (x < 0.6) & (y > -0.80) & (y < -0.07)
    x_min, x_max = 0.33, 0.6
    y_min, y_max = -0.80, -0.07
    z_min, z_max = 0.0, 0.5
    
    # 生成框线的顶点
    corners = [
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],  # 底面
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],  # 顶面
    ]
    
    # 添加框线
    bbox_lines = []
    # 底面边
    for i in range(4):
        bbox_lines.extend([corners[i], corners[(i+1)%4]])
    # 顶面边  
    for i in range(4, 8):
        bbox_lines.extend([corners[i], corners[4 + (i-4+1)%4]])
    # 竖直边
    for i in range(4):
        bbox_lines.extend([corners[i], corners[i+4]])
    
    server.scene.add_point_cloud(
        'bbox_1_frame',
        np.array(bbox_lines),
        point_size=0.002,
        point_shape="circle",
        colors=(255, 255, 0)  # 黄色边框
    )
    
    # 第二个bounding box: (z > 0.03) & (z < 0.055) & (x > 0.325) & (x < 0.64) & (y > -0.825) & (y < 0.4175)
    x_min2, x_max2 = 0.325, 0.64
    y_min2, y_max2 = -0.825, 0.4175
    z_min2, z_max2 = 0.03, 0.055
    
    corners2 = [
        [x_min2, y_min2, z_min2], [x_max2, y_min2, z_min2], [x_max2, y_max2, z_min2], [x_min2, y_max2, z_min2],
        [x_min2, y_min2, z_max2], [x_max2, y_min2, z_max2], [x_max2, y_max2, z_max2], [x_min2, y_max2, z_max2],
    ]
    
    bbox_lines2 = []
    for i in range(4):
        bbox_lines2.extend([corners2[i], corners2[(i+1)%4]])
    for i in range(4, 8):
        bbox_lines2.extend([corners2[i], corners2[4 + (i-4+1)%4]])
    for i in range(4):
        bbox_lines2.extend([corners2[i], corners2[i+4]])
    
    server.scene.add_point_cloud(
        'bbox_2_frame',
        np.array(bbox_lines2),
        point_size=0.002,
        point_shape="circle",
        colors=(0, 255, 255)  # 青色边框
    )


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    
    # 添加世界坐标系网格（只需要添加一次）
    add_world_grid(server)

    while True:
        frame = camera.get_frame(require_pc=True)
        print('timestamp:', frame['timestamp'])

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
            point_size=0.001,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        z_bounding_lower, z_bounding_upper = 0, 0.2
        z_table_lower, z_table_upper = 0.03, 0.055
        server.scene.add_point_cloud(
            'pc_marker',
            np.array([
                [0.325, 0.175, z_bounding_lower],
                [0.825, -0.325, z_bounding_lower],
                [0.325, -0.825, z_bounding_lower],
                [-0.175, -0.325, z_bounding_lower],
                [0.325, 0.175, z_bounding_upper],
                [0.825, -0.325, z_bounding_upper],
                [0.325, -0.825, z_bounding_upper],
                [-0.175, -0.325, z_bounding_upper],
                [0.325, 0.175, z_table_lower],
                [0.825, -0.325, z_table_lower],
                [0.325, -0.825, z_table_lower],
                [-0.175, -0.325, z_table_lower],
                [0.325, 0.175, z_table_upper],
                [0.825, -0.325, z_table_upper],
                [0.325, -0.825, z_table_upper],
                [-0.175, -0.325, z_table_upper],
            ]),
            point_size=0.003,
            point_shape="circle",
            colors=(255, 0, 0)
        )