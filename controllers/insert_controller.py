from typing import List, Optional
import numpy as np

import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction

class InsertController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: RMPFlowController,
        gripper: ParallelGripper,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
    
        BaseController.__init__(self, name=name)
        self._events_dt = events_dt
        
        if self._events_dt is None:
            self._events_dt = [0.05,0.004,0.004,0.004,0.05,0.005,0.005,0.008,0.08,0.05,0.05]
        self._event = -1
        self._t = 0
        self._cspace_controller = cspace_controller
        self._gripper = gripper
       
        return
    
    def forward(
        self,
        picking_position: np.ndarray,
        picking_orientation: np.ndarray,
        
        end_effect_orientation: np.ndarray, 
        
        insert_position: np.ndarray,
        insert_orientation : np.ndarray
        
    ) -> ArticulationAction:
        
        ##그리퍼 open
        # print(self._event)
        if self._event==-1:
            target_joint_positions = self._gripper.forward(action="open")
            
        if self._event==0:
            picking_position=np.array([picking_position[0],picking_position[1],0.4])
            target_joint_positions = self._cspace_controller.forward(target_end_effector_position=picking_position,
                                                                target_end_effector_orientation=None)
        ## 그립 포지션 위에 위치
        elif self._event==1:
            picking_position=np.array([picking_position[0],picking_position[1],0.4])
            target_joint_positions = self._cspace_controller.forward(target_end_effector_position=picking_position,
                                                                target_end_effector_orientation=picking_orientation)
            
         ## 그립 포지션 내려감
        elif self._event==2:
            # picking_position=np.array([picking_position[0],picking_position[1],0.5])
            target_joint_positions = self._cspace_controller.forward(target_end_effector_position=picking_position,
                                                                target_end_effector_orientation=picking_orientation)
        ## 그립퍼 close
        elif self._event==3:
                target_joint_positions = self._gripper.forward(action="close")
                
        elif self._event==4:
                picking_position = np.array([0.45,0.2,0.5])
                
                target_joint_positions = self._cspace_controller.forward(target_end_effector_position=picking_position)
                                                                        #  target_end_effector_orientation=insert_orientation)
                # needle은 미리 돌려야함 insert_orientation 원래는 ori 변수 자체가 없음
        elif self._event==5:
                insert_position = np.array([insert_position[0],insert_position[1],0.5])
                target_joint_positions = self._cspace_controller.forward(target_end_effector_position=insert_position,
                                                                    target_end_effector_orientation=picking_orientation)
                # cuboid는 picking_orientation
                # syl 은  insert_orientation
        
        elif self._event==6:
            insert_position = np.array([insert_position[0],insert_position[1],0.5])
            target_joint_positions = self._cspace_controller.forward(target_end_effector_position=insert_position,
                                                                    target_end_effector_orientation=insert_orientation)
        
        elif self._event==7:
            target_joint_positions = self._cspace_controller.forward(target_end_effector_position=insert_position,
                                                                    target_end_effector_orientation=insert_orientation)        
        elif self._event==8:
            target_joint_positions = self._gripper.forward(action="open")
        
        elif self._event == 9:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=np.array([0.2, -0.4, 0.5]),
                target_end_effector_orientation=None
            )   
            
        self._t += self._events_dt[self._event]
        
        if self._t >= 1.8:
            # 1.4, 1.5 
            self._event += 1
            self._t = 0
        
        return target_joint_positions
    
    def reset(self) -> None:
        self._event = -1  # 초기화 시 open 상태부터 시작
        self._t = 0

    def is_done(self) -> bool:
        # _event가 마지막 상태를 초과하면 True 반환
        return self._event > 9