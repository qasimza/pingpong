import torch
import time
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from env.tasks.humanoid_amp import HumanoidAMP
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidPPGetup(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob = cfg["env"]["fallInitProb"]

        self._reset_fall_env_ids = []

        self.move_xy = True

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)


        self._tar_pos = torch.ones([self.num_envs, 2], device=self.device, dtype=torch.float)

        self._ball_init_pos = torch.tensor([0, 0, 1.4], device=self.device, dtype=torch.float)

        self._ball_init_vel_weight = torch.tensor([0.4,2], device=self.device, dtype=torch.float)
        self._ball_init_vel_bias = torch.tensor([-3,-1], device=self.device, dtype=torch.float)

        # if (not self.headless):
        #     self._build_table_state_tensors()
        
        # self._build_ball_state_tensors()
        self._generate_fall_states()
        
        

        return

    

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self._update_recovery_count()
        return
    
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        
        self._table_handles = []
        self._ball_handles = []
        self._load_table_asset()
        self._load_ball_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_table_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "table.urdf"

        asset_options = gymapi.AssetOptions()
        # asset_options.max_angular_velocity = 100.0
        # asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.01
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._table_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # prp = self.gym.get_asset_rigid_shape_properties(self._table_asset)
        # for i in range(len(prp)):
        #     prp[i].restitution = 0.5
        #     prp[i].thickness = 0.0
        #     prp[i].contact_offset = 0.0
        #     prp[i].friction = 0.0
        # self.gym.set_asset_rigid_shape_properties(self._table_asset,prp)
        return

    def _load_ball_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "ball.urdf"

        asset_options = gymapi.AssetOptions()

        asset_options.max_angular_velocity = 30.0

        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        for each in prp:
            each.restitution = 0.8
            # each.thickness = 0.001
            each.contact_offset = 0.02
            each.friction = 1.
            each.rest_offset = 0.01
            # each.rolling_friction = 1
            
        self.gym.set_asset_rigid_shape_properties(self._ball_asset,prp)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if (not self.headless):
        # self._build_table(env_id, env_ptr)
        # self._build_ball(env_id, env_ptr)

        return

    # def _build_table(self, env_id, env_ptr):
    #     col_group = env_id
    #     col_filter = 2
    #     segmentation_id = 1
    #     default_pose = gymapi.Transform()
        
    #     table_handle = self.gym.create_actor(env_ptr, self._table_asset, default_pose, "table", col_group, col_filter, segmentation_id)
        
    #     rsp = gymapi.RigidShapeProperties()
    #     rsp.restitution = 0.5
    #     rsp.thickness = 0.2
    #     rsp.contact_offset = 0.2
    #     rsp.friction = 0.8

    #     self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, [rsp,rsp])

    #     # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 0.41, 0.88))
    #     self._table_handles.append(table_handle)

        return

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

        return

    # def _build_ball_state_tensors(self):
    #     num_actors = self._root_states.shape[0] // self.num_envs
    #     self._ball_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
    #     self._ball_pos = self._ball_states[..., :3]
    #     self._ball_vel = self._ball_states[..., 7:10]
        
    #     self._ball_actor_ids = self._humanoid_actor_ids + 2

    #     return
    
    # def _reset_ball(self, env_ids):
    #     if len(env_ids)>0:
            
    #         self._ball_pos[env_ids, 0:3] = self._ball_init_pos.clone()
    #         # self._ball_pos[env_ids, 2] = 2

    #         self._ball_vel[env_ids, :2] = torch.rand([len(env_ids),2], device=self.device, dtype=torch.float) * self._ball_init_vel_weight + self._ball_init_vel_bias
    #         # print(self._ball_pos[env_ids, 0:3])
    #         # print(self._ball_vel[env_ids, :2])
    #         self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
    #                                                     gymtorch.unwrap_tensor(self._ball_actor_ids), len(self._ball_actor_ids))
    #     return

    def _generate_fall_states(self):
        max_steps = 150
        
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        

        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states
        
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            # time.sleep(0.1)
            self.gym.simulate(self.sim)
            
        self._refresh_sim_tensors()
        
        self._fall_root_states = self._humanoid_root_states.clone()
        self._fall_root_states[:, 7:13] = 0
        self._fall_dof_pos = self._dof_pos.clone()
        self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        return

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask)

        recovery_ids = env_ids[recovery_mask]
        if (len(recovery_ids) > 0):
            self._reset_recovery_episode(recovery_ids)
            

        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        fall_probs = to_torch(np.array([self._fall_init_prob] * nonrecovery_ids.shape[0]), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if (len(fall_ids) > 0):
            self._reset_fall_episode(fall_ids)
            

        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if (len(nonfall_ids) > 0):
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return
    
    def _reset_fall_episode(self, env_ids):
        fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
        self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids
        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)
        # self._reset_ball(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if (len(self._reset_fall_env_ids) > 0):
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        return

    
