"""Whole Body Tracking command presets for the T1-29dof robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="",
    motion_dir="motions_t1_29dof_npz",

    body_names_to_track=[
        "Trunk",
        "Waist",
        "Shank_Left",
        "left_foot_link",
        "Shank_Right",
        "right_foot_link",
        "AL3",
        "left_hand_link",
        "AR3",
        "right_hand_link",
    ],

    body_name_ref=["Trunk"],

    use_adaptive_timesteps_sampler=False,
    use_adaptive_motion_sampler=True,
    noise_to_initial_pose=init_pose_config,
)

t1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")
    },
)

__all__ = ["t1_29dof_wbt_command"]
