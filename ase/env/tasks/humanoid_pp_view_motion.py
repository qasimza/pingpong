import torch

from isaacgym import gymtorch

from env.tasks.humanoid_amp import HumanoidAMP
from isaacgym import gymapi
import numpy as np

class HumanoidPPViewMotion(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt
        self.move_xy = False
        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        # self._build_ball_state_tensors()

        return

    # def _load_ball_asset(self):
    #     asset_root = "ase/data/assets/mjcf/"
    #     asset_file = "ball.urdf"

    #     asset_options = gymapi.AssetOptions()

    #     asset_options.max_angular_velocity = 30.0

    #     asset_options.fix_base_link = False
    #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    #     self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    #     prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
    #     for each in prp:
    #         each.restitution = 0.8
    #         # each.thickness = 0.001
    #         each.contact_offset = 0.02
    #         each.friction = 1.5
    #         # each.rest_offset = 0.01
    #         # each.rolling_friction = 1
            
    #     self.gym.set_asset_rigid_shape_properties(self._ball_asset,prp)

    #     return

    # def _build_ball(self, env_id, env_ptr):
    #     col_group = env_id
    #     col_filter = 3
    #     segmentation_id = 2
    #     default_pose = gymapi.Transform()

        

    #     ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "pingpong_ball", col_group, col_filter, segmentation_id)
    #     # rsp = gymapi.RigidShapeProperties()

    #     # rsp.restitution = 0.8
    #     # rsp.thickness = 0.
    #     # rsp.contact_offset = 0.
    #     # rsp.friction = 0.
    #     # self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, [rsp])
    #     self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.5, 0.5))
    #     self._ball_handles.append(ball_handle)

    #     return

    def _create_envs(self, num_envs, spacing, num_per_row):
        
        # self._table_handles = []
        # self._ball_handles = []
        # self._load_table_asset()
        # self._load_ball_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if (not self.headless):
        # self._build_table(env_id, env_ptr)
        # self._build_ball(env_id, env_ptr)

        return

    # def _build_ball_state_tensors(self):
    #     num_actors = self._root_states.shape[0] // self.num_envs
    #     self._ball_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
    #     self._ball_pos = self._ball_states[..., :3]
    #     self._ball_vel = self._ball_states[..., 7:10]
        
    #     self._ball_actor_ids = self._humanoid_actor_ids + 1

    #     return
    
    # def _reset_ball(self, env_ids):
    #     if len(env_ids)>0:
            
    #         self._ball_pos[env_ids, 0:3] = self._ball_init_pos.clone()
    #         self._ball_pos[0, 2] += 0.5

    #         self._ball_vel[env_ids, :2] = torch.rand([len(env_ids),2], device=self.device, dtype=torch.float) * self._ball_init_vel_weight + self._ball_init_vel_bias
    #         # print(self._ball_pos[env_ids, 0:3])
    #         # print(self._ball_vel[env_ids, :2])
    #         self._ball_vel[0,0] = -2.6
    #         self._ball_vel[0,1] = 0.85

    #         self._ball_pos[0,0] = 1
    #         self._ball_vel[0,2] = 0.7
    #         # self._ball_vel[0,1] = 0

    #         self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
    #                                                     gymtorch.unwrap_tensor(self._ball_actor_ids), len(self._ball_actor_ids))
    #     return

    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)
        # self._reset_ball(env_ids)
        return
    
    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = torch.zeros_like(self.actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        return

    def post_physics_step(self):
        
        if self.save_bvh:
            root_pos = self._humanoid_root_states[..., 0:3].clone()
            root_rot = self._humanoid_root_states[..., 3:7].clone()

            self.record_qpos.append(np.concatenate([root_pos[0].cpu().tolist(), root_rot[0].cpu().tolist(), self._dof_pos[0].cpu().tolist()]))

            self.record_body_pos.append(self._rigid_body_pos[0].cpu().tolist())
            self.record_body_rot.append(self._rigid_body_rot[0].cpu().tolist())
        
        if self.save_bvh and self.frame_count > self.save_frames:
            np.savez(
                f"saved_results/{self.rgb_path}/bvh_data.npz",
                qpos=np.array(self.record_qpos),
                pos=np.array(self.record_body_pos),
                rot=np.array(self.record_body_rot)
            )
            
        super().post_physics_step()
        self._motion_sync()
        return
    
    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_actors(self, env_ids):
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated