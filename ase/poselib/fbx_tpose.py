import torch
import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
fbx_file = "data/01_01_cmu.fbx"

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=2
)
tpose = SkeletonState.zero_pose(motion.skeleton_tree)
first_frame = motion.tensor[0, :]
tpose.tensor = first_frame
translation = tpose.root_translation
# translation += torch.tensor([0, 0, 34])
# save motion in npy format
# motion.to_file("data/01_01_cmu.npy")
# tpose.to_file("data/try_tpose.npy")
# visualize motion

plot_skeleton_state(tpose)