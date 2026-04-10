import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
_HERE = os.path.dirname(__file__)
fbx_file = os.path.join(_HERE, "data", "pingpong", "backhand_blender.fbx")

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=120
)

# save motion in npy format
motion.to_file(os.path.join(_HERE, "data", "pingpong", "backhand_blender.npy"))

# visualize motion
plot_skeleton_motion_interactive(motion)
