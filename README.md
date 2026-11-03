# Booster T1 Humanoid: Hybrid Reinforcement Learning

A hybrid Reinforcement Learning and task-level control framework for the **Booster T1 humanoid robot**, developed and evaluated in the MuJoCo physics simulator.

## Overview

This project explores reinforcement learning for humanoid locomotion, balance, and coordinated manipulation using the Booster T1 robot model. It combines a Proximal Policy Optimization (PPO) controller, a pretrained locomotion policy, and task-level motion planning to execute an integrated pick-and-place task.

The implementation focuses on combining learned control with conventional robotics techniques to achieve stable locomotion and coordinated object transfer in simulation.

## Key Features

- **Reinforcement Learning:** PPO-based training and evaluation for humanoid balance and stability.
- **Locomotion Control:** Integration of a pretrained Booster T1 locomotion policy for commanded movement and waypoint navigation.
- **Inverse Kinematics:** Numerical inverse kinematics using SciPy to generate right-arm joint configurations.
- **Coordinated Manipulation:** Scripted arm trajectories for reaching, lifting, carrying, and placing objects.
- **Integrated Pick-and-Place:** Simulated object transfer between two tables, including navigation along a predefined route.
- **Performance Evaluation:** Analysis of balance, torso stability, locomotion, joint behaviour, and task execution.

## Control Architecture

The system uses a hybrid control approach:

| Component | Method |
|---|---|
| Balance and stability | PPO (Stable-Baselines3) |
| Locomotion | Pretrained Booster T1 policy |
| Arm motion | Numerical inverse kinematics |
| Motion execution | Interpolated joint trajectories |
| Object attachment | MuJoCo mocap-based kinematic attachment |
| Simulation | MuJoCo |

The PPO and pretrained locomotion controllers are used in different phases of the integrated task. Task-level logic coordinates controller transitions and manipulation actions.

## Integrated Demonstration

The final simulation demonstrates the following sequence:

1. Humanoid approaches the pickup table.
2. Transitions to PPO-based balance control.
3. Reaches towards the object using inverse kinematics.
4. Attaches the object to the hand using a MuJoCo mocap-based constraint.
5. Transitions back to the pretrained locomotion controller.
6. Navigates along a predefined route towards the destination table.
7. Positions and releases the object at the target location.
8. Retracts the arm to its home configuration.

## Technologies Used

- Python
- MuJoCo
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- SciPy
- YAML

## Project Scope and Limitations

This implementation is a **hybrid RL and task-level control system**, rather than an end-to-end reinforcement learning solution.

- PPO is used for balance and stability control.
- Locomotion relies on a pretrained Booster T1 policy.
- Arm trajectories are generated using numerical inverse kinematics and scripted interpolation.
- Object attachment and release are kinematic operations in simulation, not physics-based grasping with contact forces.

These design choices provide a framework for studying the integration of learned control and conventional robotics methods in humanoid tasks.

## Future Scope

- End-to-end or hierarchical reinforcement learning for locomotion and manipulation.
- Physics-based grasping and contact-aware manipulation.
- Dynamic obstacle avoidance and autonomous route planning.
- Sim-to-real transfer and deployment on physical humanoid hardware.

---

**Project:** Reinforcement Learning for Robotics  
**Robot:** Booster T1 Humanoid  
**Simulation:** MuJoCo