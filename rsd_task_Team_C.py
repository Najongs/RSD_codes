import os
import sys
import cv2
import numpy as np
import time

sys.path.append("/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/exts")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from panda.isaac.tasks.pick_insertion_task import PickInsertion
from omni.isaac.franka import KinematicsSolver
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from matplotlib import pyplot as plt
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka.controllers import InsertController
from omni.isaac.franka.controllers import OverlapController
from omni.isaac.franka.controllers import XY_OverlapController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController


# Constants
SEG_COLORS = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]), np.array([255, 255, 0])]
SAVE_ROOT = os.path.join(
    "/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka", "images"
)
os.makedirs(SAVE_ROOT, exist_ok=True)

def save_image(rgb, depth, file_name):
    min_depth, max_depth = 1.9, depth.max()
    depth = (1 - (depth - min_depth) / (max_depth - min_depth)) * 255
    depth = depth.astype('uint8')

    cv2.imwrite(file_name + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name + "_depth.png", depth)
    print("Images saved successfully.")

# World setting
my_world = World(physics_dt= 0.01,stage_units_in_meters=1.0)
my_task = PickInsertion()
my_world.add_task(my_task)
my_world.reset()

# Get instance
task_params = my_task.get_params()
franka_name = my_task.get_params()["robot_name"]["value"]
camera_name = my_task.get_params()["camera_name"]["value"]

my_franka = my_world.scene.get_object(franka_name)
my_camera = my_world.scene.get_object(camera_name)

my_franka_kin = KinematicsSolver(my_franka) 

cspace_controller = RMPFlowController(name="panda_controller",robot_articulation=my_franka)
overlap_controller = OverlapController(name="overlap_controller", cspace_controller=cspace_controller, gripper=my_franka.gripper)
xy_overlap_controller = XY_OverlapController(name="xy_overlap_controller", cspace_controller=cspace_controller, gripper=my_franka.gripper)
insert_controller = InsertController(name="insert_controller",cspace_controller=cspace_controller,gripper=my_franka.gripper)

articulation_controller = my_franka.get_articulation_controller()

my_world.reset()

t = 0.0
camera_state = 0
reset_needed = False
observations = None

print("Simulation starting...")

while simulation_app.is_running():
    my_world.step(render=True)
    
    # Handle simulation stopping
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    # Handle simulation reset
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = True

        # Capture images
        rgb_image = my_camera.get_rgba()
        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        
        # Step 1: Move end effector to initial position
        if t < 2:
            actions = cspace_controller.forward(
                target_end_effector_position=np.array([0.2, -0.4, 0.5]),
                target_end_effector_orientation=None,)
            articulation_controller.apply_action(actions)

        # Step 2: Save image and process observations
        if camera_state == 0 and t > 2:
            save_image(rgb_image, distance_image, os.path.join(SAVE_ROOT, "Test"))
            observations = my_world.get_observations()
            print(f"Current event state: {observations['state']['event_state']}")
            camera_state = 1
            
        # Step 3: Check z_overlap and perform pick-and-place
        if t > 2.5 and observations["state"]["event_state"] == 0:
            if overlap_controller.is_done():
                overlap_controller.reset()
                camera_state = 0
            else:
                name_list = ["cylinder", "hexagonal_prism", "needle", "torus", "tube", "cuboid"]
                num = next((i for i, value in enumerate(observations["state"]["z_overlap"]) if value == 1), None)
                object_name = name_list[num]

                actions = overlap_controller.forward(
                    picking_position=observations[task_params[object_name]["value"]]["position"],
                    picking_orientation=observations[task_params[object_name]["value"]]["orientation"],
                    blank_position=observations["state"]["empty_space"],
                    blank_orientation=None,
                )
                articulation_controller.apply_action(actions)
                    
        # Step 3.5: Perform insertion tasks
        if t > 2.5 and observations["state"]["event_state"] == 1:
            if overlap_controller.is_done():
                overlap_controller.reset()
                camera_state = 0         
            else:
                # Determine object for insertion
                name_list = ["cuboid", "cylinder", "tube", "hexagonal_prism", "needle", "torus"]
                for object_name, adjusted_pos in observations["state"]["xy_overlap"]: 
                    break

                actions = overlap_controller.forward(
                    picking_position=observations[task_params[object_name]["value"]]["position"],
                    picking_orientation=observations[task_params[object_name]["value"]]["orientation"],
                    blank_position=observations["state"]["empty_space"],
                    blank_orientation=None,
                )
                articulation_controller.apply_action(actions)
                
        # Step 4: Perform insertion tasks
        if t > 2.5 and observations["state"]["event_state"] == 2:
            if insert_controller.is_done():
                insert_controller.reset()
                camera_state = 0
            else:
                # Determine object for insertion
                name_list = ["cuboid", "cylinder", "tube", "hexagonal_prism", "needle", "torus"]
                num = 0
                object_name = name_list[num]

                actions = insert_controller.forward(
                    picking_position=observations[task_params[object_name]["value"]]["position"],
                    picking_orientation=observations[task_params[object_name]["value"]]["orientation"],
                    end_effect_orientation=observations[task_params["robot_name"]["value"]]["end_effect_orientation"],
                    insert_position=observations[task_params[object_name]["value"]]["goal_position"],
                    insert_orientation=observations[task_params[object_name]["value"]]["goal_orientation"],
                )
                articulation_controller.apply_action(actions)

        t += my_world.get_physics_dt()

simulation_app.close()
