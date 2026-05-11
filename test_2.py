from isaaclab.app import AppLauncher

# ------------------------------------------------
# Launch Isaac Sim
# ------------------------------------------------

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ------------------------------------------------
# Imports
# ------------------------------------------------

import torch
from policy import PolicyNetwork
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim.spawners.from_files import UsdFileCfg, GroundPlaneCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ------------------------------------------------
# Create Simulation
# ------------------------------------------------

sim_cfg = SimulationCfg(dt=0.005)
sim = SimulationContext(sim_cfg)

# ------------------------------------------------
# Ground Plane
# ------------------------------------------------

GroundPlaneCfg().func(
    "/World/defaultGroundPlane",
    GroundPlaneCfg(),
)

# ------------------------------------------------
# Robot Config
# ------------------------------------------------

robot_cfg = ArticulationCfg(

    prim_path="/World/G1",

    spawn=UsdFileCfg(
        usd_path="/home/ril/Downloads/robot_1.usd",
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=10.0,
        )
    }
)

# ------------------------------------------------
# Create Robot
# ------------------------------------------------

robot = Articulation(robot_cfg)

# ------------------------------------------------
# Reset Simulation
# ------------------------------------------------

sim.reset()
obs_dim = 77
action_dim = 29

policy = PolicyNetwork(obs_dim, action_dim).cuda()

print("Robot loaded successfully!")

# Print body names once
print(robot.body_names)

# =========================================================
# BODY IDS
# =========================================================

left_foot_name = "left_ankle_roll_link"
right_foot_name = "right_ankle_roll_link"

head_name = "head_link"
pelvis_name = "pelvis"

left_foot_id = robot.body_names.index(left_foot_name)
right_foot_id = robot.body_names.index(right_foot_name)

head_id = robot.body_names.index(head_name)
pelvis_id = robot.body_names.index(pelvis_name)

# ------------------------------------------------
# Main Loop
# ------------------------------------------------

while simulation_app.is_running():

    # ------------------------------------------------
    # Step physics
    # ------------------------------------------------

    sim.step()

    # ------------------------------------------------
    # Update robot
    # ------------------------------------------------

    robot.update(sim.get_physics_dt())

    # =========================================================
    # BASIC STATES
    # =========================================================

    # Joint states
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    # Root / pelvis states
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w

    root_lin_vel = robot.data.root_lin_vel_w
    root_ang_vel = robot.data.root_ang_vel_w

    # =========================================================
    # COM FEATURES
    # =========================================================

    # COM forward velocity
    com_vel_x = root_lin_vel[:, 0:1]

    # COM height
    com_height = root_pos[:, 2:3]

    # Pelvis angular velocity
    pelvis_ang_vel = root_ang_vel

    # =========================================================
    # PHYSX LINK FORCES
    # =========================================================

    link_incoming_forces = robot.root_physx_view.get_link_incoming_joint_force()

    left_foot_force = link_incoming_forces[:, left_foot_id, 0:3]
    right_foot_force = link_incoming_forces[:, right_foot_id, 0:3]

    # =========================================================
    # BODY POSITIONS
    # =========================================================

    body_pos = robot.data.body_pos_w

    head_pos = body_pos[:, head_id, :]
    pelvis_pos = body_pos[:, pelvis_id, :]

    # =========================================================
    # FOOT VELOCITIES
    # =========================================================

    body_lin_vel = robot.data.body_lin_vel_w

    left_foot_vel = body_lin_vel[:, left_foot_id, :]
    right_foot_vel = body_lin_vel[:, right_foot_id, :]

    # Speed magnitude
    left_foot_speed = torch.norm(left_foot_vel, dim=-1, keepdim=True)
    right_foot_speed = torch.norm(right_foot_vel, dim=-1, keepdim=True)

    # =========================================================
    # FINAL OBSERVATION VECTOR
    # =========================================================

    obs = torch.cat(
        [
            # Joint information
            joint_pos,
            joint_vel,

            # Foot forces
            left_foot_force,
            right_foot_force,

            # COM information
            com_vel_x,
            com_height,

            # Pelvis angular velocity
            pelvis_ang_vel,

            # Body positions
            head_pos,
            pelvis_pos,

            # Foot speeds
            left_foot_speed,
            right_foot_speed,
        ],
        dim=-1
    )

    # =========================================================
    # POLICY NETWORK
    # =========================================================

    with torch.no_grad():

        actions = policy(obs)

    # =========================================================
    # APPLY ACTIONS
    # =========================================================

    robot.set_joint_position_target(actions)

    # =========================================================
    # PRINT INFO
    # =========================================================

    print("Observation Shape:", obs.shape)
    print("Actions Shape:", actions.shape)

    print("Left Foot Force:", left_foot_force)
    print("Right Foot Force:", right_foot_force)
    # ------------------------------------------------
    # Print Info
    # ------------------------------------------------

    print("Left Foot Force:", left_foot_force)
    print("Right Foot Force:", right_foot_force)

    print("Joint Position Shape:", joint_pos.shape)
    print("Joint Velocity Shape:", joint_vel.shape)

    print("Observation Shape:", obs.shape)
    

# ------------------------------------------------
# Close App
# ------------------------------------------------

simulation_app.close()
