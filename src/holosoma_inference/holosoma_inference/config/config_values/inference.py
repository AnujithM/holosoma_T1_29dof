"""Default inference configurations for holosoma_inference."""

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

# Shared safety secondary for all G1 configs — FastSAC locomotion.
# Each config references the same object; users can override any field
# with --secondary.task.model-path etc., or disable with --secondary none.
_g1_safety_secondary = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.safety_locomotion_g1,
)

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=replace(task.locomotion, model_path=task.safety_locomotion_g1.model_path),
    secondary=_g1_safety_secondary,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
_g1_29dof_wbt_robot = replace(
    robot.g1_29dof,
    stiff_startup_pos=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                          # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
    ),
    stiff_startup_kp=(
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ),
)

g1_29dof_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_robot,
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
    secondary=_g1_safety_secondary,
)

# T1 Whole-Body Tracking (23DOF)
t1_23dof_wbt = InferenceConfig(
    robot=replace(
        robot.t1_23dof,
        stiff_startup_pos=(
            0.0,
            0.0,  # head (yaw, pitch)
            0.2,
            -1.35,
            0.0,
            -0.5,  # left arm
            0.2,
            1.35,
            0.0,
            0.5,  # right arm
            0.0,  # waist
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,  # left leg
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,  # right leg
        ),
        stiff_startup_kp=(
            20,
            20,  # head
            20,
            20,
            20,
            20,  # left arm
            20,
            20,
            20,
            20,  # right arm
            200,  # waist
            200,
            200,
            200,
            200,
            50,
            50,  # left leg
            200,
            200,
            200,
            200,
            50,
            50,  # right leg
        ),
        stiff_startup_kd=(
            0.2,
            0.2,  # head
            0.5,
            0.5,
            0.5,
            0.5,  # left arm
            0.5,
            0.5,
            0.5,
            0.5,  # right arm
            5,  # waist
            5,
            5,
            5,
            5,
            3,
            3,  # left leg
            5,
            5,
            5,
            5,
            3,
            3,  # right leg
        ),
    ),
    observation=observation.wbt_23dof,
    task=task.wbt,
)

# T1 Whole-Body Tracking (29DOF)
# fmt: off
_t1_29dof_wbt_robot = replace(
    robot.t1_29dof,
    stiff_startup_pos=(
        0.0, 0.0,                                          # head (yaw, pitch)
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,             # left arm (7 DOF)
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,               # right arm (7 DOF)
        0.0,                                                # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,                   # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,                   # right leg
    ),
    stiff_startup_kp=(
        20, 20,                                             # head
        20, 20, 20, 20, 20, 20, 20,                         # left arm
        20, 20, 20, 20, 20, 20, 20,                         # right arm
        200,                                                # waist
        200, 200, 200, 200, 50, 50,                          # left leg
        200, 200, 200, 200, 50, 50,                          # right leg
    ),
    stiff_startup_kd=(
        0.2, 0.2,                                           # head
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,                 # left arm
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,                 # right arm
        5,                                                  # waist
        5, 5, 5, 5, 3, 3,                                    # left leg
        5, 5, 5, 5, 3, 3,                                    # right leg
    ),
)
# fmt: on

t1_29dof_wbt = InferenceConfig(
    robot=_t1_29dof_wbt_robot,
    observation=observation.wbt,
    task=task.wbt,
)

# Core defaults - no extension imports at module load time
DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "t1-23dof-wbt": t1_23dof_wbt,
    "t1-29dof-wbt": t1_29dof_wbt,
}

# Track whether extensions have been loaded
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.inference"):
        DEFAULTS[ep.name] = ep.load()


def get_annotated_inference_config() -> type:
    """Build the annotated InferenceConfig type with all discovered configs.

    This function loads extension configs lazily and returns a tyro-compatible
    annotated type for CLI subcommand generation.

    Returns:
        Annotated type suitable for use with tyro.cli()
    """
    _load_extensions()
    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in DEFAULTS.items()}
            )
        ),
    ]


def get_defaults() -> dict:
    """Get all inference config defaults, including extensions.

    Returns:
        Dictionary mapping config names to InferenceConfig instances.
    """
    _load_extensions()
    return DEFAULTS
