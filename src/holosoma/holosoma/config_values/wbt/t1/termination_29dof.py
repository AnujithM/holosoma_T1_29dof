"""Whole Body Tracking termination presets for the T1-29dof robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

t1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": [
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
                "bad_motion_body_pos_body_names": [
                    "left_foot_link",
                    "right_foot_link",
                    "left_hand_link",
                    "right_hand_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["t1_29dof_wbt_termination"]
