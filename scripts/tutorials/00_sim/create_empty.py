# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse # Standard Python module for passing in cmd line args

from isaaclab.app import AppLauncher #AppLauncher is a ultility class from isaac lab library, acts as a wrapper for the sim app 

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.") # Creating instance of ArgumentParser class inside of module argparse.
# append AppLauncher cli args
# adds_app_launcher_arg method of AppLauncher class that adds cmd line args used by isaac sim (headless, livestream, off-screen rendering) to ArgumentParser instance
AppLauncher.add_app_launcher_args(parser) 
# parse the cmd line arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli) #Passing in parsed arguments into an instance of the AppLauncher class
simulation_app = app_launcher.app #Launching the simulation app with the configuration of app_launcher

"""Rest everything follows."""

#Simulation must run before importing these!
#The module isaaclab.sim is the goat (spawn obj, modify USD prims, and converts 3D files into USD)
#SimulationContext is a class that controles sim related events (phys, rendering)
#SimulationCfg is a class that configures simulation phys
from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01) #Setting up the simulation configuration w/ physics and a timestep we are defining
    sim = SimulationContext(sim_cfg) #Initilazing the simulation w/ the simulation configs above
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0]) #[camera's position in 3D space], [target pt the cam should look at]

    # Play the simulator
    sim.reset() #w/ the simulation scene set up, we have to initiazle the phys handles before stepping
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running(): #Reset method automatically starts simulations so this starts off true
        # perform step
        sim.step() #steps sim, takes in arg render to determine whether rendering events are updated


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
