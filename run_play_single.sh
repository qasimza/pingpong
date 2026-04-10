python ase/run.py --test --num_envs 2 --task HumanoidPPHitTargetsSingle \
--cfg_env ase/data/cfg/humanoid_pp.yaml \
--cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file ase/data/motions/pingpong/Transition_looping_FB_blender_pph.npy \
--llc_checkpoint output/pp_llc_forehand/nn/Humanoid.pth \
--checkpoint output/pp_hlc_forehand/nn/Humanoid.pth --skill forehand