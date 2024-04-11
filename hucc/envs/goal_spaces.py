# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


# Delta features for standard joint observations (not goal-space delta_feats!)
# this feature of the regular "joints" observation are removed before passing to
# the low-level policy.
g_delta_feats = {
    "Walker": [1],
    "Humanoid": [0, 1],
    "HumanoidPC": [0, 1],
    "Reacher": [],
    "Franka": [],  # proprioceptive is just qpos/qvel
    "WAM": [],  # proprioceptive is just qpos/qvel
}


# TODO: why does index g[0] exist if it seems to be dropped?
# TODO: add def_ranges with many inputs such that each stores its delta/twist which are concatenated
def def_ranges(
    inp: List[Tuple],
    delta_feats: List[int] = None,
    twist_feats: List[int] = None,
) -> Dict[str, List]:
    return {
        "str": [g[1] for g in inp],
        "min": [g[2] for g in inp],
        "max": [g[3] for g in inp],
        "delta_feats": delta_feats if delta_feats else [],
        "twist_feats": twist_feats if twist_feats else [],
    }


g_goal_ranges_bodyfeet_walker: List[Tuple] = [
    (0, "rootz:p", +0.95, +1.50),  # 0.9 is fall-over
    (1, "rootx:p", -3.00, +3.00),
    (2, "rooty:p", -1.30, +1.30),  # 1.4 is fall-over
    (3, "left_foot:px", -0.72, +0.99),
    (4, "left_foot:pz", -1.30, +0.00),
    (5, "right_foot:px", -0.72, +0.99),
    (6, "right_foot:pz", -1.30, +0.00),
]

g_goal_ranges_bodyfeet_humanoid: List[Tuple] = [
    (0, "root:px", -3.00, +3.00),
    (1, "root:py", -3.00, +3.00),  # we can rotate so equalize this
    (2, "root:pz", +0.95, +1.50),  # 0.9 is falling over
    (3, "root:tz", -1.57, +1.57),  # stay within +- pi/2, i.e. don't turn around
    (4, "root:sy", -1.57, +1.57),  # Swing around Y axis
    (5, "root:sx", -0.50, +0.50),  # Swing around X axis (more or less random v)
    (6, "left_foot:px", -1.00, +1.00),
    (7, "left_foot:py", -1.00, +1.00),
    (8, "left_foot:pz", -1.00, +0.20),
    (9, "right_foot:px", -1.00, +1.00),
    (10, "right_foot:py", -1.00, +1.00),
    (11, "right_foot:pz", -1.00, +0.20),
]

# franka dt = 2ms, frame_skip = 10, step = 20ms, 100 steps -> 2s
g_goal_ranges_fingerpos_franka: List[Tuple] = [
    # box position bound
    (0, "tipx", +0.20, +0.70),
    (1, "tipy", -0.55, +0.55),
    (2, "tipz", +0.01, +0.20),
]

g_goal_ranges_fingerposboxz_franka: List[Tuple] = [
    # limit z into box height,
    # box bottom is 0.02 thick, rest on plane at -0.02 -> min 0
    # box side are 0.09 high, start 0.0035 + 0.01 above bottom -> max -0.02 + 0.0135 + 0.09 = 0.0835
    # some slack -> z in [0.01, 0.07]
    (0, "tipz", +0.01, +0.07),
]

g_goal_ranges_fingervel_franka: List[Tuple] = [
    # covered x,y distance < 0.3, 0.9 (even random_init), 2s time => 0.6/s, 1.8/s enough
    # covered z distance < 0.2 (above box and in), 2s time => 0.4/s enough
    (0, "tipx_vel", -0.6, +0.6),
    (1, "tipy_vel", -1.8, +1.8),
    (2, "tipz_vel", -0.4, +0.4),
]

g_goal_ranges_fingerpos_delta_franka: List[Tuple] = [
    # speed 0.6,1.8,0.4 /s enough, 20ms step, 0.012,0.036,0.008/step, interval 5 => 0.06,0.18,0.04
    (0, "tipx_delta", -0.06, +0.06),
    (1, "tipy_delta", -0.18, +0.18),
    (2, "tipz_delta", -0.04, +0.04),
]

g_goal_ranges_fingereuler_franka: List[Tuple] = [
    # extrinsic xyz, scipy.spatial.transform.Rotation.as_euler
    # TODO: restrict ranges to allowed cone, this way most goals unreachable,
    # because they are considered invalid.
    # NOTE: cant use quaternions, as no normalization after random sampling and
    # difference l2 is not the geodesic distance of quaternions (similar problems
    # with gimbal lock of euler angles as well)
    (0, "eulerx", -1.0 * np.pi, +1.0 * np.pi),
    (1, "eulery", -0.5 * np.pi, +0.5 * np.pi),
    (2, "eulerz", -1.0 * np.pi, +1.0 * np.pi),
]

g_goal_ranges_fingerswingtwist_franka: List[Tuple] = [
    # decompose finger rotation into twist around z, then swing around axis in xy plane
    # twist angle is "from above" in xy plane, swing gives "tilt" of rod.
    # unrotated rod points up, so inital vertical rod (with tip below) has 180deg swing
    # target swing in 180+-30 deg for upright
    # NOTE: twist is undefined if rod vertical, i.e. swing 180deg, defaults to 0 twist,
    # but twist angle is consistent else e.g. for swing 179 and 181
    (0, "swingxy", np.pi - np.pi / 6, np.pi + np.pi / 6),
    (1, "twistz", +0.0 * np.pi, +2.0 * np.pi),
]

# reacher dt = 10ms, frame_skip = 2, step=20ms, 200 steps -> 4s
g_goal_ranges_joint_value_reacher: List[Tuple] = [
    # cover full 360 degree range
    (0, "joint0", -1.0 * np.pi, +1.0 * np.pi),
    (1, "joint1", -1.0 * np.pi, +1.0 * np.pi),
    (2, "joint2", -1.0 * np.pi, +1.0 * np.pi),
    (3, "joint3", -1.0 * np.pi, +1.0 * np.pi),
    (4, "joint4", -1.0 * np.pi, +1.0 * np.pi),
]

g_goal_ranges_joint_vel_reacher: List[Tuple] = [
    # distance from reset < pi, 4s time => pi/s enough
    (0, "vel0", -1.0 * np.pi, +1.0 * np.pi),
    (1, "vel1", -1.0 * np.pi, +1.0 * np.pi),
    (2, "vel2", -1.0 * np.pi, +1.0 * np.pi),
    (3, "vel3", -1.0 * np.pi, +1.0 * np.pi),
    (4, "vel4", -1.0 * np.pi, +1.0 * np.pi),
]

g_goal_ranges_joint_value_delta_reacher: List[Tuple] = [
    # speed pi/s enough, 20ms step, 0.063/step, interval 5 => 0.1pi
    (0, "joint0_delta", -0.1 * np.pi, +0.1 * np.pi),
    (1, "joint1_delta", -0.1 * np.pi, +0.1 * np.pi),
    (2, "joint2_delta", -0.1 * np.pi, +0.1 * np.pi),
    (3, "joint3_delta", -0.1 * np.pi, +0.1 * np.pi),
    (4, "joint4_delta", -0.1 * np.pi, +0.1 * np.pi),
]
g_goal_ranges_fingerpos_reacher: List[Tuple] = [
    # corners of box constraint are unreachable but entire reach covered
    (0, "tipx", -0.5, +0.5),
    (1, "tipy", -0.5, +0.5),
]

g_goal_ranges_fingervel_reacher: List[Tuple] = [
    # distance from reset < 1, 4s time => 1/s enough
    (0, "tipx_vel", -1.0, +1.0),
    (1, "tipy_vel", -1.0, +1.0),
]

g_goal_ranges_fingerpos_delta_reacher: List[Tuple] = [
    # speed 1/s enough, 20ms step, 0.02/step, interval 5 => 0.1
    (0, "tipx_delta", -0.1, +0.1),
    (1, "tipy_delta", -0.1, +0.1),
]


# tabletennis dt = 2ms, ~600 sim steps till hit -> 1.2s; frame_skip = 4 -> step = 8ms, ~150 steps till hit
# table is [-.7625, .7625] wide in y dir, at height z~=0.76, ends at x=1.37
# barret wam shoulder joint is at x=2.1, y=0, z = 1.654, can reach up to radius 1.1 (fully extended) below ~30 deg upwards
# ball has inital x-vel of 2.5
g_goal_ranges_fingervel_tabletennis: List[Tuple] = [
    # 3/4 spin from forward around back to side of hit in less than 1.2s
    # 3/4 * 2*pi * .7625 = 3.6, 5.2 / 1.2 = 3
    # considering acceleration, set x and y vel range to +-5
    # z only needs to cover 1.5 (radius 1.1 90 deg down to ~30 deg up) -> range 2
    # covered x,y distance < 0.3, 0.9 (even random_init), 2s time => 0.6/s, 1.8/s enough
    # covered z distance < 0.2 (above box and in), 2s time => 0.4/s enough
    (0, "tipx_vel", -5.0, +5.0),
    (1, "tipy_vel", -5.0, +5.0),
    (2, "tipz_vel", -2.0, +2.0),
]

g_goal_ranges_fingerpos_delta_tabletennis: List[Tuple] = [
    # speed 5,5,2 /s enough, 8ms step, 0.04,0.04,0.016/step, interval 5 => 0.2,0.2,0.08
    (0, "tipx_delta", -0.20, +0.20),
    (1, "tipy_delta", -0.20, +0.20),
    (2, "tipz_delta", -0.08, +0.08),
]

g_goal_ranges_fingeryawpitch_tabletennis: List[Tuple] = [
    # decompose bat normal (local x) direction into angle to global x around z (yaw) and
    # angle to global xy plane (pitch).
    # support both backhand and forehand and arbitrary pitch, so both +- 180deg
    (0, "yaw", -np.pi, np.pi),
    (1, "pitch", -np.pi, +np.pi),
]


g_goal_ranges_fingervel_delta_tabletennis: List[Tuple] = [
    # be able to accelerate to vel limits in < 150/2 steps, say 50
    # 5 / 50 = 0.1 m / s*step, 2 / 50 = 0.04 m / s*step
    # interval 5 => 0.5, 0.5, 0.2
    (0, "tipx_vel_delta", -0.5, +0.5),
    (1, "tipy_vel_delta", -0.5, +0.5),
    (2, "tipz_vel_delta", -0.2, +0.2),
]


g_goal_ranges_fingeryawpitch_delta_tabletennis: List[Tuple] = [
    # +- pi in 1.2s => ~3 rad/s, considering acceleration 5 rad/s
    # 8ms step, interval 5 => 0.2
    (0, "yaw_delta", -0.2, 0.2),
    (1, "pitch_delta", -0.2, +0.2),
]

g_goal_spaces_bodyfeet: Dict[str, Dict[str, List]] = {
    "Walker": def_ranges(g_goal_ranges_bodyfeet_walker, [1]),
    "Humanoid": def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
    "HumanoidPC": def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
}

g_goal_spaces_fingerpos: Dict[str, Dict[str, List]] = {
    "Franka": def_ranges(g_goal_ranges_fingerpos_franka),
}

g_goal_spaces_fingerpos_delta: Dict[str, Dict[str, List]] = {
    # delta x,y,z relative to current rod tip position
    "Franka": def_ranges(g_goal_ranges_fingerpos_delta_franka, [0, 1, 2]),
}


g_goal_spaces_fingerpos_fingerposdelta_fingervel_fingereuler_fingerswingtwist_fingerposboxz: Dict[
    str, Dict[str, List]
] = {
    "Franka": def_ranges(
        g_goal_ranges_fingerpos_franka  # 3
        + g_goal_ranges_fingerpos_delta_franka  # 3 delta
        + g_goal_ranges_fingervel_franka  # 3
        + g_goal_ranges_fingereuler_franka  # 3 twist
        + g_goal_ranges_fingerswingtwist_franka  # 2 twist
        + g_goal_ranges_fingerposboxz_franka,  # 1
        delta_feats=[3, 4, 5],
        twist_feats=[9, 10, 11, 12, 13],
    ),
}

g_goal_spaces_fingerposdelta_fingervel_fingeryawpitch: Dict[str, Dict[str, List]] = {
    "WAM": def_ranges(
        g_goal_ranges_fingerpos_delta_tabletennis  # 3 delta
        + g_goal_ranges_fingervel_tabletennis  # 3
        + g_goal_ranges_fingeryawpitch_tabletennis,  # 2 twist
        delta_feats=[0, 1, 2],
        twist_feats=[6, 7],
    ),
}

g_goal_spaces_fingerposdelta_fingerveldelta_fingeryawpitchdelta: Dict[str, Dict[str, List]] = {
    "WAM": def_ranges(
        g_goal_ranges_fingerpos_delta_tabletennis  # 3 delta
        + g_goal_ranges_fingervel_delta_tabletennis  # 3 delta
        + g_goal_ranges_fingeryawpitch_delta_tabletennis,  # 2 delta twist
        delta_feats=[0, 1, 2, 3, 4, 5, 6, 7],
        twist_feats=[6, 7],
    ),
}


g_goal_spaces_joint_value: Dict[str, Dict[str, List]] = {
    "Reacher": def_ranges(g_goal_ranges_joint_value_reacher),
}

g_goal_spaces_joint_value_vel: Dict[str, Dict[str, List]] = {
    "Reacher": def_ranges(
        g_goal_ranges_joint_value_reacher + g_goal_ranges_joint_vel_reacher
    ),
}


g_goal_spaces_jointvalue_jointvaluedelta_jointvel_fingerpos_fingerposdelta_fingervel: Dict[
    str, Dict[str, List]
] = {
    "Reacher": def_ranges(
        g_goal_ranges_joint_value_reacher  # 5
        + g_goal_ranges_joint_value_delta_reacher  # 5 delta
        + g_goal_ranges_joint_vel_reacher  # 5
        + g_goal_ranges_fingerpos_reacher  # 2
        + g_goal_ranges_fingerpos_delta_reacher  # 2 delta
        + g_goal_ranges_fingervel_reacher,  # 2
        delta_feats=[5, 6, 7, 8, 9, 17, 18],
    ),
}

g_goal_spaces: Dict[str, Dict[str, Dict[str, List]]] = {
    "bodyfeet": g_goal_spaces_bodyfeet,
    "bodyfeet-relz": g_goal_spaces_bodyfeet,
    "fingerpos": g_goal_spaces_fingerpos,
    "fingerpos_delta": g_goal_spaces_fingerpos_delta,
    "fpos_fposd_fvel_feuler_fswitwi_fposboxz": g_goal_spaces_fingerpos_fingerposdelta_fingervel_fingereuler_fingerswingtwist_fingerposboxz,
    "joint_value": g_goal_spaces_joint_value,
    "joint_value_vel": g_goal_spaces_joint_value_vel,
    "jval_jvald_jvel_fpos_fposd_fvel": g_goal_spaces_jointvalue_jointvaluedelta_jointvel_fingerpos_fingerposdelta_fingervel,
    "fposd_fvel_fyawpitch": g_goal_spaces_fingerposdelta_fingervel_fingeryawpitch,
    "fposd_fveld_fyawpitchd": g_goal_spaces_fingerposdelta_fingerveldelta_fingeryawpitchdelta,
}


def subsets_task_map(
    features: str, robot: str, spec: str, rank_min: int, rank_max: int
):
    """
    Parses a spec of features and returns the necessary data to construct goal
    spaces.
    Spec can be any of the following:
    - 'all': all features
    - 'torso': everything involving the robot's torso/root
    - #-separated list (elements can be feature combinations, i.e. 1+2 or 1-2)
    - a regex matching feature names

    Uncontrollable features will be removed. What's returned is a list of
    feature subsets, each entry in the form "0,1,3+4" (i.e. comma-separated,
    with combinations retained) and a task map that maps each feature to an
    index.
    """
    gs = g_goal_spaces[features][robot]
    n = len(gs["str"])
    dims: List[str] = []
    if spec == "all":
        dims = [str(i) for i in range(n)]
    elif spec == "torso":
        dims = [
            str(i)
            for i in range(n)
            if gs["str"][i].startswith(":")
            or gs["str"][i].startswith("torso:")
            or gs["str"][i].startswith("root")
        ]
    else:
        try:
            dims = []
            for d in str(spec).split("#"):
                ds = sorted(map(int, re.split("[-+]", d)))
                dims.append("+".join(map(str, ds)))
        except:
            dims = [
                str(i) for i in range(n) if re.match(spec, gs["str"][i]) is not None
            ]

    def is_controllable(d: int):
        if d < 0:
            raise ValueError(f"Feature {d} out of range")
        if d >= len(gs["min"]):
            return False
        # Return whether range is non-zero
        return gs["min"][d] != gs["max"][d]

    uncontrollable = set()
    for dim in dims:
        for idx in map(int, dim.split("+")):
            if not is_controllable(idx):
                uncontrollable.add(dim)
                log.warning(f"Removing uncontrollable feature {dim}")
                break
    dims = [dim for dim in dims if not dim in uncontrollable]
    if len(dims) < rank_min:
        raise ValueError("Less features to control than the requested rank")

    udims: Set[int] = set()
    for dim in dims:
        for idx in map(int, dim.split("+")):
            udims.add(idx)
    task_map: Dict[str, int] = {}
    for idx in sorted(udims):
        task_map[str(idx)] = len(task_map)

    def unify(comb) -> str:
        udims: Set[str] = set()
        for c in comb:
            for d in c.split("+"):
                if d in udims:
                    raise ValueError(f"Overlapping feature dimensions: {comb}")
                udims.add(d)
        return ",".join(sorted(comb, key=lambda x: [int(i) for i in x.split("+")]))

    if rank_min > 0 and rank_max > 0:
        cdims = []
        for r in range(rank_min, rank_max + 1):
            for comb in combinations(dims, r):
                # XXX Duplications are ok now
                # cdims.append(unify(comb))
                cdims.append(",".join(comb))
        dims = cdims

    return dims, task_map
