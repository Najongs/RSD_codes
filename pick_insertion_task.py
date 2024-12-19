from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid, create_prim, delete_prim
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import create_prim, delete_prim, define_prim, get_prim_path
from omni.isaac.franka import Franka
from omni.isaac.cortex.cortex_utils import get_assets_root_path
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import Camera
from pxr import UsdPhysics,Sdf
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from panda.isaac.tasks.point_convert import get_object_center_and_angle
from panda.isaac.tasks.z_pos_detection import process_image_and_depth
from panda.isaac.tasks.find_blank_pos import find_empty_space, detect_objects
from panda.isaac.tasks.xy_object_move import resolve_overlap_with_direction, get_object_center


from omni.physx.scripts import utils
import omni.usd
from omni.usd import get_context
import cv2
import omni.kit.commands

from typing import Optional
from scipy.spatial.transform import Rotation
import numpy as np
import os



class PickInsertion(BaseTask):
    def __init__(self, name: str = "pick_insertion", 
                 offset: Optional[np.ndarray] = None, 
) -> None:
        self.stage = get_context().get_stage()
        BaseTask.__init__(self, name=name, offset=offset)
        self._fr3 = None
        self._holes = []
        self._objects = []
        self.object_hsv_ranges = {  "cuboid": (0, 6, 131, 255, 32, 233),
                                    "hexagonal_prism": (85, 111, 202, 255, 67, 242),
                                    "needle": (127, 173, 46, 255, 139, 255),
                                    "torus": (9, 48, 252, 255, 1, 255),
                                    "tube": (46, 88, 134, 255, 75, 255),
                                    "cylinder": (114, 139, 173, 255, 75, 255)}
        self._scaling_list = [0.0004, 0.00043396, 0.0004, 0.0004, 0.00043333, 0.00039474]
        self._goal_position = np.array([[0.5+0.2,-0.43, 0.35],
                                       [0.5+0.2, -0.3, 0.35],
                                       [0.5+0.2, -0.17, 0.35],
                                       [0.37+0.2, -0.43, 0.35],
                                       [0.37+0.2, -0.3, 0.35],
                                       [0.37+0.2, -0.17, 0.35]])
        self._task_event = 0
        self._task_achieved = False
        self.first = True
        filepath = os.path.abspath(__file__)
        split = filepath.split('/')
        root_path = ''
        for name in split[1:9]:
            root_path = root_path + '/' + name
        self._assets_root_path = root_path        
        self._fr3_asset_name = "fr3.usd"
        self._object_asset_name = ["cylinder", "hexagonal_prism", "needle", "torus", "tube", "cuboid"]
        self._hole_asset_name = ["hole_" + name for name in self._object_asset_name]
        self.stage = omni.usd.get_context().get_stage()
        return

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()        

        franka_name = f"franka_0"
        franka_prim_path = f"/World/robots/{franka_name}"
        self._fr3 = scene.add(Franka(prim_path=franka_prim_path,name=franka_name,position=[0.0, 0.0, 0.0] + self._offset))

        for i in range(6):
            translation = np.zeros(3)
            translation[0] = np.random.uniform(low=-0.2, high=0.2)     # 20cm variation
            translation[1] = np.random.uniform(low=-0.3, high=0.3)     # 30cm variation
            translation += np.array([0.4, 0.3, 0.2])   # drop objects from 20cm height
            orientation = np.random.uniform(low=0.0, high=1.0, size=4)
            orientation = orientation / np.linalg.norm(orientation)
            object_asset_path = self._assets_root_path + "/panda/isaac/objects/" + self._object_asset_name[i] + ".usd"            
            add_reference_to_stage(usd_path=object_asset_path, prim_path="/World/objects")            
            self._objects.append(
                scene.add(
                    RigidPrim(
                        prim_path="/World/objects/" + self._object_asset_name[i],
                        name=self._object_asset_name[i],
                        translation=translation,
                        orientation=orientation
                        )
                    )
                )
            prim_path="/World/objects/" + self._object_asset_name[i]+"/"+self._object_asset_name[i]+"/tn__jgj0av6/Mesh"
            prim = self.stage.GetPrimAtPath(prim_path)
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            collision_attr = collision_api.CreateApproximationAttr()
            collision_attr.Set("convexDecomposition")
            
            prim_path="/World/objects/" + self._object_asset_name[i]+"/Looks/Diffuse"
            prim = self.stage.GetPrimAtPath(prim_path)
            material = UsdPhysics.MaterialAPI.Apply(prim)
            material.CreateDynamicFrictionAttr(1)
            material.CreateStaticFrictionAttr(1)
            material.CreateRestitutionAttr(0.1)

        index = 0        
        for i in range(2):
            for j in range(3):                
                hole_asset_path = self._assets_root_path + "/panda/isaac/objects/" + self._hole_asset_name[index] + ".usd"
                add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/objects")
                self._holes.append(
                    scene.add(
                        XFormPrim(
                            prim_path="/World/objects/" + self._hole_asset_name[index],
                            name=self._hole_asset_name[index],
                            translation=[0.5-0.13*i+0.2, -0.3-0.13+0.13*j, 0.0],
                            orientation=[1.0, 0.0, 0.0, 0.0]
                            )
                        )
                    )
                index += 1
        
        for i in range(6):
            add_update_semantics(prim=self._objects[i].prim, semantic_label=self._object_asset_name[i])
            add_update_semantics(prim=self._holes[i].prim, semantic_label=self._hole_asset_name[i])
            
        # Camera setting
        self._camera = scene.add(
            Camera(
                prim_path="/World/camera",
                name="cam_0",
                frequency=30,
                resolution=(1280, 720),
                )
            )    
        ori = Rotation.from_euler('z', angles=-90, degrees=True).as_quat() # (x,y,z,w) in scipy
        ori = ori[[3, 0, 1, 2]] # (w,x,y,z) in Isaac scipy
        self._camera.set_world_pose(position=[0.5, 0.0, 2.0], orientation=ori, camera_axes='usd')        
        self._camera.initialize()
        self._camera.set_focal_length(2)
        self._camera.set_focus_distance(1.8) 
        self._camera.add_distance_to_image_plane_to_frame()
        
        return
    
    def reset_objects(self):
        for i, obj in enumerate(self._objects):
            translation = np.zeros(3)
            translation[0] = np.random.uniform(low=-0.2, high=0.2)     # X축 변동
            translation[1] = np.random.uniform(low=-0.3, high=0.3)     # Y축 변동
            translation += np.array([0.4, 0.3, 0.2])   # 기본 오프셋 (테이블 상부 정도로 가정)

            orientation = np.random.uniform(low=0.0, high=1.0, size=4)
            orientation = orientation / np.linalg.norm(orientation)

            obj.set_world_pose(translation, orientation)
        return

    def get_observations(self) -> dict:
        # 실시간 추론을 위한 코드
        # if self.first == True:
        #     rgb = self._camera.get_rgba()
        #     rgb = (rgb * 255).astype(np.uint8)
        #     image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
        #     current_frame = self._camera.get_current_frame()
        #     depth = current_frame["distance_to_image_plane"]
            
        #     min_depth, max_depth = 1.9, depth.max()
        #     depth = (1 - (depth - min_depth) / (max_depth - min_depth)) * 255
        #     depth = depth.astype('uint8')

        #     cv2.imwrite("Test_rgb.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite("Test_depth.png", depth)
        #     print("Images saved successfully.")
        #     self.first = False
            
        image_path = '/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka/images/Test_rgb.png'
        depth_image_path = '/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka/images/Test_depth.png'
        image = cv2.imread(image_path)
        distance_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        
        # 초기화
        object_pose = []
        object_ori = []
        object_heights = [0.05, 0.06676, 0.01772, 0.02, 0.0754, 0.03]
        z_overlap_list = []
        xy_overlap_list = []
        
        # 로봇 초기 상태 가져오기
        joint_state_0 = self._fr3.get_joints_state()
        position_0, orientation_0 = self._fr3.end_effector.get_local_pose()

        # 각 객체의 포즈 계산 및 z_overlap 확인
        for i in range(6):
            # 객체의 중심과 각도 탐지
            position, orientation = get_object_center_and_angle(image, self._objects[i].name)
            
            # 깊이 정보 처리
            _, z_position = process_image_and_depth(
                image, distance_image, self._objects[i].name, self._scaling_list[i]
            )
            
            # 월드 좌표로 변환
            if z_position >= 0.07:
                position = np.array([position[1], -position[0], z_position - 0.04])
            elif z_position <= 0.02:
                position = np.array([position[1], -position[0], 0.02])
            else:
                position = np.array([position[1], -position[0], z_position/2 -0.0003])
                # cuboid 보정계수 0.0003
                # cyl 보정계수 0.0020
                
            orientation = euler_angles_to_quat(np.array([0, np.pi, np.pi / -180 * orientation]))
                
            object_pose.append(position)
            object_ori.append(orientation)

            # z_overlap 판단
            if z_position > object_heights[i] + 0.01:
                z_overlap_list.append(1)
            else:
                z_overlap_list.append(0)
                
        if 1 not in z_overlap_list:
                self._task_event = 1
        else: 
            self._task_event = 0
            
        print(f"Z Overlap List: {z_overlap_list}")

        # xy overlap 확인
        object_positions = []
        object_names = list(self.object_hsv_ranges.keys())

        # 객체 위치 추출
        for obj_name in object_names:
            position = get_object_center(image, obj_name)
            if position is not None:  # 유효한 위치만 추가
                object_positions.append(position)

        # 겹침 해결
        if len(object_positions) > 1:
            adjusted_positions = resolve_overlap_with_direction(object_positions, min_dist_threshold=0.1, move_distance=0.05)
            for i, adjusted_pos in enumerate(adjusted_positions):
                if not np.array_equal(object_positions[i], adjusted_pos):
                    xy_overlap_list.append([self._objects[i].name, adjusted_pos])
                    
        if 1 not in z_overlap_list and len(xy_overlap_list) == 0:
                self._task_event = 2
                
        print(f"XY Overlap List: {xy_overlap_list}")
        
        # 빈 공간 탐색
        detected_objects = detect_objects(image)
        try:
            _, empty_x, empty_y = find_empty_space(
                detected_objects, grid_size=(540, 720), offset=200, resolution=10, min_distance_threshold=0.1
            )
            empty_space = np.array([empty_x, empty_y, 0.2])
            
        except ValueError as e:
            print(f"Error finding empty space: {e}")
            empty_space = np.array([0, 0, 0])
            
            
        # 관측 결과 반환
        observations = {
            "state": {
                "event_state": self._task_event,
                "z_overlap": z_overlap_list,
                "xy_overlap": xy_overlap_list,
                "empty_space": empty_space,
            },
            self._fr3.name: {
                "joint_positions": np.array(joint_state_0.positions),
                "end_effect_orientation": np.array(orientation_0),
            },
        }

        # 각 객체의 상태 추가
        for i in range(6):
            observations[self._objects[i].name] = {
                "position": object_pose[i],
                "orientation": object_ori[i],
                "goal_position": self._goal_position[i],
                # 기존값 
                "goal_orientation": euler_angles_to_quat(np.array([np.pi/2 , -np.pi/2, 0]))
                if i >= 2 else euler_angles_to_quat(np.array([0, np.pi / 2, 0])),
                # needle 보정 계수 np.pi, 0, 0 (모양과 반대로 있어야함)
                # "goal_orientation": euler_angles_to_quat(np.array([np.pi, 0, 0]))
            }

        return observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        BaseTask.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        
        if time_step_index % 200 == 0:  
            print(
                f"스텝 인덱스: {time_step_index}, "
                f"시뮬레이션 타임: {simulation_time:.2f}s, "
                f"현재 상태: {self._task_event}"
            )
        return

    def post_reset(self) -> None:
        self._task_achieved = False
        return

    def cleanup(self) -> None:
        return

    def get_params(self) -> dict:
        params_representation = {
            "robot_name": {"value": self._fr3.name, "modifiable": False},
            "camera_name": {"value": self._camera.name, "modifiable": False},
            "cylinder": {"value": self._objects[0].name, "modifiable": False},
            "hexagonal_prism": {"value": self._objects[1].name, "modifiable": False},
            "needle": {"value": self._objects[2].name, "modifiable": False},
            "torus": {"value": self._objects[3].name, "modifiable": False},
            "tube": {"value": self._objects[4].name, "modifiable": False},
            "cuboid": {"value": self._objects[5].name, "modifiable": False},
            "task_event": {"value": self._task_event, "modifiable": False},
        }

        return params_representation


      