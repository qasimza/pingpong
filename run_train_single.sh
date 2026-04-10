python ase/run.py --task HumanoidPPHitTargetsSingle \
--cfg_env ase/data/cfg/humanoid_pp.yaml \
--cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file ase/data/motions/pingpong/Transition_looping_FB_blender_pph.npy \
--llc_checkpoint output/pp_llc_backhand/nn/Humanoid.pth --headless --skill backhand