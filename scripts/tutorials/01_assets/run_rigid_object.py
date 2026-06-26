# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first.""" #Tutorial 1


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows.""" #Imports about launching simulator

import torch #tensor computations for pos, states, etc

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils #math for sim
from isaaclab.assets import RigidObject, RigidObjectCfg #RigidObject class: rigid obj w/ physical properties and statemanagement. RigidObjectCfg class: coresponding config
from isaaclab.sim import SimulationContext


def design_scene(): 
    """Designs the scene."""
    #------Tutorial 2------
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    #---------------------

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]] #list of origins we want create x-forms at
    for i, origin in enumerate(origins): #iterating loop to create x-forms at each origin
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object, doing so w/o func method (this way of doing things better for complex scenes)
    cone_cfg = RigidObjectCfg( #Spawning cfg is done so w/ RigidObjectCfg class
        prim_path="/World/Origin.*/Cone", #Everypath that World that starts w/ Origin
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(), #Identify pose: pos, rotation, linear & angular velocities all 0
    )
    cone_object = RigidObject(cfg=cone_cfg) #Instead of func method, spawns 

    # return the scene information
    scene_entities = {"cone": cone_object}
    return scene_entities, origins #so main can access the cone_object, done with InteractiveScene class instead of a dict later on


#Overview: Handles simualtion loop, resets obj states, steps the simulation, & updates data buffers
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"] #Extract the cone object from the entities dictionary
    # Define simulation stepping
    sim_dt = sim.get_physics_dt() #timestep set to default of 0.017, 60hz
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0: #reset every 250 steps
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = cone_object.data.default_root_state.clone() #Get default root state: a tensor [num_instances, 13] which includes rigid body pos, rotation, lienar velocity, and angular velocity
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins #[all rows (cones), first 3 columns (xyz pos)], shifts each cone to its assigned origin (local frame --> world frame)
            root_state[:, :3] += math_utils.sample_cylinder( #Random cylindrical offset
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device 
            )
            # write root state to simulation
            cone_object.write_root_pose_to_sim(root_state[:, :7]) #To update w/ obj's new state
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            cone_object.reset() #ensure obj start w/ newly defined state, clearing any internal data
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        cone_object.write_data_to_sim() #Writes things like external force (not in this tutorial) to the simulation buffer
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt) #updating state of obj's in simulation
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_pos_w}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene() #Design the scene with the specified entities and origins
    scene_origins = torch.tensor(scene_origins, device=sim.device) #Make the origins list to a tensor living on the simulation device (GPU)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins) #Calling the run_simulator function


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
