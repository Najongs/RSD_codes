from typing import List, Optional
import numpy as np

import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction

class OverlapController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: RMPFlowController,
        gripper: ParallelGripper,
        end_effector_initial_height: Optional[float] = 0.0,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        super().__init__(name=name)
        
        # 이벤트 지속 시간 설정
        self._events_dt = events_dt or [0.05,0.004,0.004,0.004,0.005,0.008,0.008,0.1]
        
        # 초기 상태 변수
        self._event = -1  # 초기 상태
        self._t = 0  # 이벤트 타이머
        self._cspace_controller = cspace_controller  # 작업 공간 컨트롤러
        self._gripper = gripper  # 평행 그리퍼
        return
    
    def forward(
        self,
        picking_position: np.ndarray,
        picking_orientation: np.ndarray,
        blank_position: np.ndarray,
        blank_orientation: np.ndarray
    ) -> ArticulationAction:
        
        target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=None,
                target_end_effector_orientation=None,
            )

        # 이벤트 처리
        if self._event == -1:  # 그리퍼 Open
            target_joint_positions = self._gripper.forward(action="open")

        elif self._event == 0:  # 픽업 대상 상단으로 이동
            picking_position=np.array([picking_position[0],picking_position[1],0.3])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=picking_orientation,
            )

        elif self._event == 1:  # 픽업 대상 근처로 이동
            picking_position=np.array([picking_position[0],picking_position[1],0.2])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=picking_orientation,
            )

        elif self._event == 2:  # 정확히 픽업 위치로 이동
            picking_position=np.array([picking_position[0],picking_position[1],picking_position[2]+0.0075])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=None,
            )

        elif self._event == 3:  # 픽업 위치에서 위로 이동
            picking_position=np.array([picking_position[0],picking_position[1],picking_position[2]])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=None,
            )

        elif self._event == 4:  # 놓을 위치로 이동
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=blank_position,
                target_end_effector_orientation=picking_orientation,
            )

        elif self._event == 5:  # 초기 위치로 복귀
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=np.array([0.2, -0.4, 0.5]),
                target_end_effector_orientation=None,
            )

        # 이벤트 진행 타이머 업데이트
        self._t += self._events_dt[self._event]
        if self._t >= 2.5:  # 이벤트 완료 시 다음 이벤트로
            self._event += 1
            self._t = 0

        return target_joint_positions

    def reset(self) -> None:
        self._event = -1  # Open 상태로 초기화
        self._t = 0

    def is_done(self) -> bool:
        return self._event > 6  # 모든 이벤트가 완료되었는지 확인
