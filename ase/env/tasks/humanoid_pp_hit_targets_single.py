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
from utils.torch_utils import *
import datetime
import os
import numpy as np


class HumanoidPPHitTargetsSingle(HumanoidAMP):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):

        self.statistics = cfg["args"].statistics

        self.move_xy = True
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        self.skill = cfg["args"].skill
        self.test = 0
        self._reset_fall_env_ids = []

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        self._recovery_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )

        # self._tar_pos = torch.ones([self.num_envs, 2], device=self.device, dtype=torch.float)

        self._build_predefined_tensors()

        self._build_ball_state_tensors()

        if not self.headless:
            self._build_target_state_tensors()
            print("####################################")

            self._build_vis_land_state_tensors()
        # if self.statistics:
        #     self._build_statistics_tensors()
        self.reset_time = 0
        return

    # def _build_statistics_tensors(self):
    #     self.avg_hits = []
    #     self.avg_ball_vel = []
    #     self.land_dis = []

    #     self.body_ang_vel = []
    #     self.body_vel = []

    #     self.hits_buf = torch.zeros(
    #         self.num_envs, device=self.device, dtype=torch.float
    #     )
    #     self.prev_predict_ball_land_pos = None

    def _build_predefined_tensors(self):
        self._ball_init_pos = torch.tensor(
            [0, 0, 1.4], device=self.device, dtype=torch.float
        )

        self.ball_launch_id = torch.bernoulli( 
            torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 0.5
        )

        self.ball_launch_strategy_prob = (
            torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 1
        )
        self.ball_launch_strategy = torch.bernoulli(self.ball_launch_strategy_prob)

        self._ball_mytable_contact_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )
        self._ball_othertable_contact_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )
        self._ball_paddle_contact_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )
        self._solid_ball_paddle_contact_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )
        self._ball_table_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int
        )

        self._ball_targets = torch.tensor(
            [1, 0, 0.8], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)

        self._ball_targets[:, 0] = torch.Tensor(self.num_envs).uniform_(0.1, 1.2)
        self._ball_targets[:, 1] = torch.Tensor(self.num_envs).uniform_(-0.65, 0.65)
        # print(self._ball_targets)
        self._ball_prev_vel_buf = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.int
        )

        self._target_root_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )

        self._target_root_pos[..., 0] = self.root_move_xy[0]  # + 0.3
        self._target_root_pos[..., 2] = 0.75

        self._target_root_transl_prob = (
            torch.zeros(self.num_envs, device=self.device, dtype=torch.float) + 0.5
        )

        self.target_root_transl = torch.bernoulli(self._target_root_transl_prob) * 0.6

        print("[[[[skill]]]]", self.skill)

        self._ball_init_vel_weight = torch.tensor(
            [0.5, 3], device=self.device, dtype=torch.float
        )
        self._ball_init_vel_bias = torch.tensor(
            [-3, -1.5], device=self.device, dtype=torch.float
        )
        if self.skill == "backhand":
            self._target_root_transl_prob = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )
            self._ball_init_vel_bias = torch.tensor(
                [-3, -2.0], device=self.device, dtype=torch.float
            )
        elif "forehand" in self.skill:
            self._target_root_transl_prob = torch.ones(
                self.num_envs, device=self.device, dtype=torch.float
            )
            self._ball_init_vel_bias = torch.tensor(
                [-3, -1.0], device=self.device, dtype=torch.float
            )

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)
        # if self.viewer and (not self.headless):
        #     self._draw_task()

        # current_time = datetime.datetime.now()
        # ct = current_time.strftime("%m-%d-%H-%M-%S")
        # image_file = f'./images/images/{ct}.png'
        # self.gym.write_camera_image_to_file(self.sim, self.envs, self.viwer_camera_handle, gymapi.IMAGE_COLOR, image_file)
        return

    def _draw_task(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._ball_pos[..., 0:3]
        ends = self._ball_targets[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            # import pdb;pdb.set_trace()
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols
            )

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 15
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._table_handles = []
        self._ball_handles = []

        self._load_table_asset()
        self._load_ball_asset()

        if not self.headless:
            self._target_handles = []
            self._load_target_asset()
            self._vis_land_handles = []
            self._load_vis_land_asset()

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

        self._table_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
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

        self._ball_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        prp = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        for each in prp:
            each.restitution = 0.8
            # each.thickness = 0.001
            each.contact_offset = 0.02
            each.friction = 1.0
            each.rest_offset = 0.01

        self.gym.set_asset_rigid_shape_properties(self._ball_asset, prp)

        return

    def _load_target_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "target.urdf"

        asset_options = gymapi.AssetOptions()

        # asset_options.max_angular_velocity = 30.0

        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        return

    def _load_vis_land_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "ball_land.urdf"

        asset_options = gymapi.AssetOptions()

        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._vis_land_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_table(env_id, env_ptr)
        self._build_ball(env_id, env_ptr)
        if not self.headless:
            self._build_target(env_id, env_ptr)
            self._build_vis_land(env_id, env_ptr)

        return

    def _build_table(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 1
        default_pose = gymapi.Transform()

        table_handle = self.gym.create_actor(
            env_ptr,
            self._table_asset,
            default_pose,
            "table",
            col_group,
            col_filter,
            segmentation_id,
        )

        rsp = gymapi.RigidShapeProperties()
        rsp.restitution = 0.7
        rsp.thickness = 0.2
        rsp.contact_offset = 0.02
        rsp.friction = 0.8
        rsp.rolling_friction = 0.1

        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, [rsp, rsp])

        self.gym.set_rigid_body_color(
            env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 0.41, 0.88)
        )
        self._table_handles.append(table_handle)

        return

    def _build_ball(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 3
        segmentation_id = 2
        default_pose = gymapi.Transform()

        ball_handle = self.gym.create_actor(
            env_ptr,
            self._ball_asset,
            default_pose,
            "pingpong_ball",
            col_group,
            col_filter,
            segmentation_id,
        )
        # rsp = gymapi.RigidShapeProperties()

        # rsp.restitution = 0.8
        # rsp.thickness = 0.
        # rsp.contact_offset = 0.
        # rsp.friction = 0.
        # self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, [rsp])
        self.gym.set_rigid_body_color(
            env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.5, 0.0)
        )
        self._ball_handles.append(ball_handle)

        return

    def _build_ball_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._ball_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 2, :]
        self._ball_pos = self._ball_states[..., :3]
        self._ball_vel = self._ball_states[..., 7:10]
        self._ball_ang_vel = self._ball_states[..., 10:13]

        self._ball_actor_ids = self._humanoid_actor_ids + 2

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # print(bodies_per_env, self.num_bodies)
        # time.sleep(10)
        self._ball_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., -2:, :]

        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 4
        segmentation_id = 3
        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(
            env_ptr,
            self._target_asset,
            default_pose,
            "target",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
        )
        self._target_handles.append(target_handle)

        return

    def _build_vis_land(self, env_id, env_ptr):
        col_group = env_id + 1
        col_filter = 5
        segmentation_id = 4
        default_pose = gymapi.Transform()

        vis_land_handle = self.gym.create_actor(
            env_ptr,
            self._vis_land_asset,
            default_pose,
            "vis_land",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, vis_land_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8)
        )
        self._vis_land_handles.append(vis_land_handle)

        return

    def _build_target_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._target_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 3, :]
        self._target_pos = self._target_states[..., :3]
        self._target_actor_ids = self._humanoid_actor_ids + 3

        return

    def _build_vis_land_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._vis_land_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 4, :]
        self._vis_land_pos = self._vis_land_states[..., :3]
        self._vis_land_actor_ids = self._humanoid_actor_ids + 4

        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if self._enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        # self._update_ball()
        return

    def post_physics_step(self):
        # print(self._ball_othertable_contact_buf[0])
        if self.save_bvh:

            ball_pos = self._ball_pos[..., :3].clone()
           
            root_pos = self._humanoid_root_states[..., 0:3].clone()
            root_rot = self._humanoid_root_states[..., 3:7].clone()

            self.record_qpos.append(np.concatenate([root_pos[0].cpu().tolist(), root_rot[0].cpu().tolist(), self._dof_pos[0].cpu().tolist()]))

            self.record_ball_pos.append(ball_pos[0].cpu().tolist())

            self.record_body_pos.append(self._rigid_body_pos[0].cpu().tolist())
            self.record_body_rot.append(self._rigid_body_rot[0].cpu().tolist())

            self.record_targets.append(self._ball_targets[0].cpu().tolist())
        
        if self.save_bvh and self.frame_count>self.save_frames:

            np.savez(f'saved_results/{self.rgb_path}/bvh_data.npz',
                ball_pos = np.array(self.record_ball_pos),
                qpos = np.array(self.record_qpos), pos = np.array(self.record_body_pos), rot = np.array(self.record_body_rot),
                targets = np.array(self.record_targets))
        
        
        
        super().post_physics_step()
        
        self._update_ball()
        

        return

    
    
    def _update_ball(self):
        #####################to do


        reset_task_mask = torch.logical_and(
            torch.logical_or(
                self._ball_pos[..., 2] < 0.7, self._ball_pos[..., 0] > 1.5
            ),
            self._ball_othertable_contact_buf >= 1,
        ) + (self._ball_othertable_contact_buf >= 2)

        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        
        
        if len(rest_env_ids) > 0:
            # self._reset_env_tensors(rest_env_ids)
            # if 0 in rest_env_ids:
            #     print(rest_env_ids[0], self.progress_buf[0])
            # self._refresh_sim_tensors()
            self._reset_task_ball(rest_env_ids)
            self._refresh_sim_tensors()
            
            # if 0 in rest_env_ids:
            #     self.flag=True
            #     import pdb;pdb.set_trace()
            
            # self._reset_ball(rest_env_ids)
            # 
            # self._compute_observations(rest_env_ids)
        return

    def _reset_env_ball(self, env_ids):
        n = len(env_ids)
        if n > 0:
            self.ball_launch_id[env_ids] = torch.bernoulli(
                torch.zeros(n, device=self.device, dtype=torch.float) + 0.5
            )

            # self.front_or_back[env_ids] = torch.bernoulli(torch.zeros(n, device=self.device, dtype=torch.float) + 0.5)

            self.ball_launch_strategy[env_ids] = torch.bernoulli(
                self.ball_launch_strategy_prob[env_ids]
            )

            self._ball_prev_vel_buf[env_ids] = 0

            self._ball_mytable_contact_buf[env_ids] = 0

            self._ball_othertable_contact_buf[env_ids] = 0

            self._ball_paddle_contact_buf[env_ids] = 0
            self._solid_ball_paddle_contact_buf[env_ids] = 0

            self._ball_table_reward_buf[env_ids] = 0

            self._ball_pos[env_ids, :3] = self._ball_init_pos.clone()
            self._ball_pos[env_ids, 0] += 0.5

            self._ball_targets[env_ids, 0] = (
                torch.Tensor(n).uniform_(0.3, 1.2).to(self.device, torch.float)
            )
            self._ball_targets[env_ids, 1] = (
                torch.Tensor(n).uniform_(-0.6, 0.6).to(self.device, torch.float)
            )

            self._ball_states[env_ids, 3:6] = 0
            self._ball_states[env_ids, 6] = 1

            self._ball_states[env_ids, 10:13] = 0

            ball_launch_pos = self._ball_pos[env_ids, :3].clone()

            if self.skill == "any":
                # self._ball_vel[env_ids, 0] = torch.Tensor(n).uniform_(-3.2, -2.6).to(self.device, torch.float)
                # self._ball_vel[env_ids, 1] = torch.Tensor(n).uniform_(-1.2, 1.2).to(self.device, torch.float)
                # self._ball_vel[env_ids, 2] = 0
                vx1, vy1, vz1 = self.get_ball_initial_speed(
                    ball_launch_pos, n, -0.5, 0.5, 2.5, 3.5, -0.6
                )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1

                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                target_root_transl = (
                    (ball_side_pos_y - 0) > 0.0
                ) * 0.6

            elif "forehand" in self.skill:
                # self._ball_vel[env_ids, 0] = torch.Tensor(n).uniform_(-2.6, -2.).to(self.device, torch.float)
                # self._ball_vel[env_ids, 1] = torch.Tensor(n).uniform_(-0., 1.2).to(self.device, torch.float)
                # self._ball_vel[env_ids, 2] = 0
                vx1, vy1, vz1 = self.get_ball_initial_speed(
                    ball_launch_pos, n, -0.0, 0.5, 2.5, 3.5, -0.6
                )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1
                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                target_root_transl = (
                    torch.bernoulli(self._target_root_transl_prob[env_ids]) * 0.6
                )

            elif self.skill == "backhand":
                # self._ball_vel[env_ids, 0] = torch.Tensor(n).uniform_(-3.2, -2.6).to(self.device, torch.float)
                # self._ball_vel[env_ids, 1] = torch.Tensor(n).uniform_(-1.2, 0.4).to(self.device, torch.float)
                # self._ball_vel[env_ids, 2] = 0

                vx1, vy1, vz1 = self.get_ball_initial_speed(
                    ball_launch_pos, n, -0.5, 0.0, 2.5, 3.5, -0.6
                )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1

                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                target_root_transl = (
                    torch.bernoulli(self._target_root_transl_prob[env_ids]) * 0.6
                )

            self.target_root_transl[env_ids] = target_root_transl.clone()
            self._target_root_pos[env_ids, 1] = (
                ball_side_pos_y.clone() - target_root_transl
            )
            # self._target_root_pos[env_ids, 0] = self.root_move_xy[0] + 0.4 * self.front_or_back[env_ids]

            # if not self.headless:
            #     self._target_pos[env_ids, :3] = self._ball_targets[env_ids, :3].clone()
            #     env_ids_int32 = torch.cat(
            #         [self._ball_actor_ids[env_ids], self._target_actor_ids[env_ids]]
            #     )
            # else:
            #     env_ids_int32 = self._ball_actor_ids[env_ids]

            # self.gym.set_actor_root_state_tensor_indexed(
            #     self.sim,
            #     gymtorch.unwrap_tensor(self._root_states),
            #     gymtorch.unwrap_tensor(env_ids_int32),
            #     len(env_ids_int32),
            # )

        return

    def get_ball_initial_speed(
        self, ball_pos, n, y_target_left, y_target_right, vx_min=2, vx_max=4, x_max=0
    ):
        # Constants
        # g = 9.81
        # net_height = 1
        # table_height = 0.8

        # x0 = 1.37
        # z0 = 1.5
        # y0 = 0
        x0 = ball_pos[:, 0]
        y0 = ball_pos[:, 1]
        z0 = ball_pos[:, 2]

        # vx = -10  # Example negative value, you can change this
        # -15 ~ -5

        vx = torch.Tensor(n).uniform_(-vx_max, -vx_min).to(self.device, torch.float)

        t_net = -x0 / vx  # abs((0 - x0) / vx)  # Time to reach the net
        # t_end = abs((-1.37 - x0) / vx)  # Time to reach the other end of the table

        vz_min = (1 - z0 + 0.5 * 9.81 * t_net**2) / t_net
        # vz_max = (table_height - z0 + 0.5 * 9.81 * t_end**2) / t_end

        a = -0.5 * 9.81
        b = vz_min
        c = z0 - 0.8
        t2 = (-b - torch.sqrt(b**2 - 4 * a * c)) / (2 * a)

        x_position_max = x0 + vx * t2
        # import pdb;pdb.set_trace()
        x_position_max = torch.clamp(x_position_max, max=x_max)

        x_target = torch.rand(n, device=self.device) * ((x_position_max) - (-1.37)) + (
            -1.37
        )
        # y_target = torch.rand(n,device=self.device) * ((0.76) - (-0.76)) + (-0.76)

        # x_target = torch.Tensor(n).uniform_(-1.37, x_position_max).to(self.device, torch.float)
        y_target = (
            torch.Tensor(n)
            .uniform_(y_target_left, y_target_right)
            .to(self.device, torch.float)
        )

        # print('####', x_target, y_target)

        t_land = (x_target - x0) / vx
        vz = (0.8 - z0 + 0.5 * 9.81 * t_land**2) / t_land
        vy = (y_target - y0) / t_land

        return vx, vy, vz

    def _reset_task_ball(self, env_ids):
        n = len(env_ids)
        # if 0 in env_ids:
        #     print('reset task!')
        #     time.sleep(0.2)
        if n > 0:
            self.ball_launch_id[env_ids] += 1

            self._ball_prev_vel_buf[env_ids] = 0

            self._ball_mytable_contact_buf[env_ids] = 0

            self._ball_othertable_contact_buf[env_ids] = 0

            self._ball_paddle_contact_buf[env_ids] = 0

            self._ball_table_reward_buf[env_ids] = 0

            # self.front_or_back[env_ids] = torch.bernoulli(torch.zeros(n, device=self.device, dtype=torch.float) + 0.5)

            self._ball_pos[env_ids, :3] = self._ball_init_pos.clone()
            self._ball_pos[env_ids, 0] += 1.37
            self._ball_pos[env_ids, 1] += (
                torch.Tensor(n).uniform_(-0.75, 0.75).to(self.device, torch.float)
            )
            self._ball_pos[env_ids, 2] += (
                torch.Tensor(n).uniform_(-0.1, 0.2).to(self.device, torch.float)
            )

            ball_launch_pos = self._ball_pos[env_ids, :3].clone()

            self._ball_targets[env_ids, 0] = (
                torch.Tensor(n).uniform_(0.1, 1.37).to(self.device, torch.float)
            )
            self._ball_targets[env_ids, 1] = (
                torch.Tensor(n).uniform_(-0.75, 0.75).to(self.device, torch.float)
            )

            # self._ball_vel[env_ids, :2] = torch.rand([n, 2], device=self.device, dtype=torch.float) * self._ball_init_vel_weight + self._ball_init_vel_bias

            # self._ball_vel[env_ids, 2] = 0

            self._ball_states[env_ids, 3:6] = 0
            self._ball_states[env_ids, 6] = 1

            self._ball_states[env_ids, 10:13] = 0

            ball_launch_id = self.ball_launch_id[env_ids]
            ball_launch_strategy = self.ball_launch_strategy[env_ids]

            if self.skill == "any":
                vx1, vy1, vz1 = self.get_ball_initial_speed(
                    ball_launch_pos, n, -0.6, 0.6, 3.5, 5
                )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1

                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                ball_side_pos_y = torch.clamp(ball_side_pos_y, min=-10, max=10)

                target_root_transl = (
                    (ball_side_pos_y - self._humanoid_root_states[env_ids, 1]) > 0.0
                ) * 0.6

            elif "forehand" in self.skill:
                if self.skill=="forehand":
                    vx1, vy1, vz1 = self.get_ball_initial_speed(
                        # ball_launch_pos, n, -0.2, 0.6, 2.5, 4, -0.4
                        ball_launch_pos, n, -0.2, 0.6, 3.5, 5
                    )
                elif self.skill=="forehand_smash":
                    vx1, vy1, vz1 = self.get_ball_initial_speed(
                        ball_launch_pos, n, -0.2, 0.6, 2.5, 4, -0.4
                        # ball_launch_pos, n, -0.2, 0.6, 3.5, 5
                    )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1

                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                ball_side_pos_y = torch.clamp(ball_side_pos_y, min=-10, max=10)

                target_root_transl = (
                    torch.bernoulli(self._target_root_transl_prob[env_ids]) * 0.6
                )

            elif self.skill == "backhand":
                vx1, vy1, vz1 = self.get_ball_initial_speed(
                    ball_launch_pos, n, -0.6, 0.2, 3.5, 5
                )
                self._ball_vel[env_ids, 0] = vx1
                self._ball_vel[env_ids, 1] = vy1
                self._ball_vel[env_ids, 2] = vz1

                ball_side_pos_y = (
                    -((1.37 + ball_launch_pos[:, 0]) / self._ball_vel[env_ids, 0])
                    * self._ball_vel[env_ids, 1]
                    + ball_launch_pos[:, 1]
                )

                ball_side_pos_y = torch.clamp(ball_side_pos_y, min=-10, max=10)

                target_root_transl = (
                    torch.bernoulli(self._target_root_transl_prob[env_ids]) * 0.6
                )

            self.target_root_transl[env_ids] = target_root_transl.clone()
            self._target_root_pos[env_ids, 1] = (
                ball_side_pos_y.clone() - target_root_transl
            )
            
            if not self.headless:
                self._target_pos[env_ids, :3] = self._ball_targets[env_ids, :3].clone()
                env_ids_int32 = torch.cat(
                    [self._ball_actor_ids[env_ids], self._target_actor_ids[env_ids]]
                )
            else:
                env_ids_int32 = self._ball_actor_ids[env_ids]

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

            

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if not self.headless:
            self._target_pos[env_ids, :3] = self._ball_targets[env_ids, :3].clone()
            env_ids_int32 = torch.cat(
                [self._ball_actor_ids[env_ids], self._target_actor_ids[env_ids]]
            )
        else:
            env_ids_int32 = self._ball_actor_ids[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []

        self._reset_env_ball(env_ids)
        # super()._reset_envs(env_ids)
        super()._reset_envs(env_ids)

        # if self.statistics:
        #     if len(env_ids) > 0:
        #         # print()
        #         avg_hits = self.hits_buf[env_ids].clone()
        #         self.avg_hits.append(avg_hits)
        #         self.hits_buf[env_ids] = 0

        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):

        if env_ids is None:
            root_states = self._humanoid_root_states.clone()
            ball_pos = self._ball_pos.clone()
            ball_vel = self._ball_vel.clone()
            target_root_pos = self._target_root_pos.clone()
            target_ball_pos = self._ball_targets.clone()

            lefthand_pos = self._rigid_body_pos[:, self._key_body_ids[1], :].clone()

            lefthand_rot = self._rigid_body_rot[:, self._key_body_ids[1], :].clone()
            leftpaddle_pos = lefthand_pos + quat_rotate(
                lefthand_rot, self._paddle_to_hand.clone()
            )

        else:
            root_states = self._humanoid_root_states[env_ids].clone()
            ball_pos = self._ball_pos[env_ids].clone()
            ball_vel = self._ball_vel[env_ids].clone()
            target_root_pos = self._target_root_pos[env_ids].clone()
            target_ball_pos = self._ball_targets[env_ids].clone()

            lefthand_pos = self._rigid_body_pos[
                env_ids, self._key_body_ids[1], :
            ].clone()
            lefthand_rot = self._rigid_body_rot[
                env_ids, self._key_body_ids[1], :
            ].clone()
            leftpaddle_pos = lefthand_pos + quat_rotate(
                lefthand_rot, self._paddle_to_hand[env_ids].clone()
            )

        obs = compute_ball_observation(
            root_states,
            ball_pos,
            ball_vel,
            target_root_pos,
            leftpaddle_pos,
            target_ball_pos,
        )
        return obs

    # def _compute_statistics(
    #     self,
    #     ball_pos,
    #     ball_vel,
    #     ball_targets,
    #     ball_paddle_contact,
    #     ball_table_contact,
    #     predict_ball_land_pos,
    # ):

    #     self.avg_ball_vel.append(torch.norm(ball_vel[ball_paddle_contact >= 1], dim=-1))

    #     # import pdb;pdb.set_trace()
    #     self.body_vel.append(torch.abs(self._rigid_body_vel).mean(dim=-1).mean(dim=-1))
    #     self.body_ang_vel.append(
    #         torch.abs(self._rigid_body_ang_vel).mean(dim=-1).mean(dim=-1)
    #     )

    #     ball_table_id = ball_table_contact == 1
    #     self.hits_buf[ball_table_id] += 1

    #     if torch.sum(ball_table_id) > 0:
    #         self.land_dis.append(
    #             torch.norm(
    #                 predict_ball_land_pos[ball_table_id, :2]
    #                 - ball_targets[ball_table_id, :2],
    #                 dim=-1,
    #             )
    #         )

    #         if len(self.avg_hits) > 1:
    #             print(
    #                 "len",
    #                 len(torch.cat(self.avg_hits[1:])),
    #                 "hits",
    #                 torch.mean(torch.cat(self.avg_hits[1:])).item(),
    #                 "land dis",
    #                 torch.mean(torch.cat(self.land_dis)).item(),
    #                 "ball vel",
    #                 torch.mean(torch.cat(self.avg_ball_vel)).item(),
    #                 "body vel",
    #                 torch.mean(torch.cat(self.body_vel)).item(),
    #                 "body ang vel",
    #                 torch.mean(torch.cat(self.body_ang_vel)).item(),
    #             )

    def _compute_reward(self, actions):

        
        

        root_pos = self._humanoid_root_states[..., 0:3].clone()
        root_rot = self._humanoid_root_states[..., 3:7].clone()

        lefthand_pos = self._rigid_body_pos[:, self._key_body_ids[1], :].clone()
        lefthand_rot = self._rigid_body_rot[:, self._key_body_ids[1], :].clone()
        

        lefthand_rot_matrix = rot_matrix_from_quaternion(lefthand_rot)

        paddle_x_dir = torch.abs(lefthand_rot_matrix[:,0,1])
        paddle_z_dir = torch.abs(lefthand_rot_matrix[:,2,1])

        
        # time.sleep(.5)
        # print(paddle_x_dir[0], paddle_z_dir[0])
        # import pdb; pdb.set_trace()
        leftpaddle_pos = lefthand_pos + quat_rotate(lefthand_rot, self._paddle_to_hand)
        # print(leftpaddle_pos[0], lefthand_pos[0])
        
        paddle_dir = lefthand_rot_matrix[:,:,1]
        ball_paddle_dir = leftpaddle_pos - self._ball_pos[...,:3]

        dot_product = torch.sum(paddle_dir * ball_paddle_dir, dim=-1)

        norm_v1 = torch.linalg.norm(paddle_dir, dim=-1)
        
        norm_v2 = torch.linalg.norm(ball_paddle_dir, dim=-1)

        # time.sleep(1)
        # print(paddle_dir[0], (ball_paddle_dir/norm_v2[:,None])[0])

        # print(dot_product / (norm_v1 * norm_v2))

        # paddle_ball_dir_cos_theta = torch.abs(dot_product / (norm_v1 * norm_v2))

        paddle_ball_dir_cos_theta = dot_product / (norm_v1 * norm_v2)

        

        height_threshold = (self._ball_pos[..., 2]<0.9)

        ball_vel_z_change = (self._ball_prev_vel_buf[...,2] * self._ball_vel[...,2])<0
        ball_vel_x_change = (self._ball_prev_vel_buf[...,0] * self._ball_vel[...,0])<0

        self._ball_prev_vel_buf = self._ball_vel.clone()

        table_x_y_threshold = torch.logical_and(torch.abs(self._ball_pos[..., 0])<1.4, torch.abs(self._ball_pos[..., 1])<0.85)
        
        contact_table = torch.logical_and(height_threshold, torch.logical_and(ball_vel_z_change, table_x_y_threshold))

        # print(contact_table[0])
        self._ball_mytable_contact_buf += 1*torch.logical_and(self._ball_pos[:,0]<0, contact_table)
        # print(self._ball_mytable_contact_buf)
        # time.sleep(0.2)

        self._ball_othertable_contact_buf += 1*torch.logical_and(self._ball_pos[:,0]>0, contact_table)
        
        ball_paddle_contact_check = torch.logical_or(ball_vel_x_change, ball_vel_z_change)
        # ball_paddle_contact_check = torch.logical_and(ball_paddle_contact_check, ball_vel_z_change)
        ball_paddle_contact_check = torch.logical_and(((torch.norm(self._ball_pos - leftpaddle_pos, dim=-1) < 0.15))*1, ball_paddle_contact_check)
        # print(ball_paddle_contact_check[0])
        self._ball_paddle_contact_buf += ball_paddle_contact_check#(torch.norm(self._ball_pos - leftpaddle_pos, dim=-1) < 0.08)*1

        if "forehand" in self.skill:
            self._solid_ball_paddle_contact_buf += torch.logical_and(paddle_ball_dir_cos_theta>0.5, ball_paddle_contact_check)

        elif self.skill == "backhand":
            self._solid_ball_paddle_contact_buf += torch.logical_and(paddle_ball_dir_cos_theta<-0.5, ball_paddle_contact_check)

        else:
            self._solid_ball_paddle_contact_buf += torch.logical_and(torch.abs(paddle_ball_dir_cos_theta)>0.5, ball_paddle_contact_check)



        # time.sleep(0.02)
        # print(self._ball_states[0,10:])
        self._ball_table_reward_buf += self._ball_othertable_contact_buf

        
        # print(self._ball_contact_forces[0])
        predict_ball_land_pos = predict_land_point(self._ball_pos[...,:3], self._ball_vel[..., :3])

        if not self.headless:
            self._update_vis_land_prediction(predict_ball_land_pos.clone(),self.num_envs)

        right_foot_pos = self._rigid_body_pos[:, self._key_body_ids[2], :].clone()

        left_foot_pos = self._rigid_body_pos[:, self._key_body_ids[3], :].clone()

        # if self.statistics:
        #     if self.prev_predict_ball_land_pos is not None:
        #         self._compute_statistics(self._ball_pos, self._ball_vel, self._ball_targets, self._ball_paddle_contact_buf, self._ball_table_reward_buf, self.prev_predict_ball_land_pos)
        #     self.prev_predict_ball_land_pos = predict_ball_land_pos.clone()

        # print(root_pos[0])

        # print(self._target_root_pos[0])

        self.rew_buf[:] = compute_ball_reward(root_pos, root_rot, 
                                                 self._ball_pos, self._ball_vel, leftpaddle_pos, self._ball_paddle_contact_buf, self._ball_table_reward_buf, self._ball_targets, self._target_root_pos.clone(), paddle_z_dir, self._solid_ball_paddle_contact_buf, predict_ball_land_pos, left_foot_pos, right_foot_pos, self._ball_ang_vel.clone())
        return

    def _update_vis_land_prediction(self, predict_pos, n):
        env_ids = list(range(n))
        self._vis_land_pos[env_ids, :2] = predict_pos[env_ids, :2]
        self._vis_land_pos[env_ids, 2:3] = 0.8
        env_ids_int32 = self._vis_land_actor_ids[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        # time.sleep(0.2)
        # self._refresh_sim_tensors()

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            self._rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights,
            self._ball_mytable_contact_buf,
            self._ball_othertable_contact_buf,
            self._ball_pos,
        )

        return


def predict_land_point(ball_pos, ball_vel):
    v_z = ball_vel[..., 2]
    z = ball_pos[..., 2]
    h = torch.abs(z - 0.8)
    t = (2 * v_z + torch.sqrt(4 * v_z**2 + 8 * 9.8 * h)) / (9.8 * 2)

    ball_land_pos = ball_pos[..., :2] + ball_vel[..., :2] * t[:, None]
    ball_land_pos = torch.clamp(ball_land_pos, min=-2, max=2)
    return ball_land_pos


def compute_ball_observation(
    root_states, ball_pos, ball_vel, target_root_pos, left_paddle_pos, target_ball_pos
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # heading_rot_ = heading_rot.clone()

    local_ball_pos = quat_rotate(heading_rot, ball_pos - root_pos)

    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    local_target_root_pos = quat_rotate(heading_rot, target_root_pos - root_pos)

    # time.sleep(0.5)
    # print(root_pos[0], target_root_pos[0], local_target_root_pos[0])

    local_paddle_pos = quat_rotate(heading_rot, left_paddle_pos - root_pos)

    local_target_ball_pos = quat_rotate(heading_rot, target_ball_pos - root_pos)

    obs = torch.cat(
        [
            local_ball_pos,
            local_ball_vel,
            local_target_root_pos,
            local_paddle_pos,
            local_target_ball_pos,
        ],
        dim=-1,
    )

    return obs


# @torch.jit.script
def compute_ball_reward(root_pos, root_rot, ball_pos, ball_vel, leftpaddle_pos, ball_paddle_contact_buf, ball_table_reward_buf, ball_targets, root_targets, paddle_z_dir, solid_ball_paddle_contact_buf, predict_ball_land_pos, left_foot_pos, right_foot_pos, ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,Tensor, Tensor) -> Tensor
    
    pos_err_scale = 4.0

    # root_targets[:, 0]+=0.4
    
    pos_diff = ball_pos - leftpaddle_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    # print(ball_vel[0])

   
    # print(ball_pos, ball_targets)
    pos_diff2 = ball_pos - ball_targets
    pos_err2 = torch.sum(pos_diff2 * pos_diff2, dim=-1)
    # pos_reward2 = torch.exp(-pos_err_scale * pos_err2)

    pos_diff3 = root_targets[:,:3] - root_pos[:,:3]
    pos_err3 = torch.sum(pos_diff3 * pos_diff3, dim=-1)
    pos_reward3 = torch.exp(-6 * pos_err3)

    pos_diff4 = predict_ball_land_pos - ball_targets[...,:2]
    pos_err4 = torch.sum(pos_diff4 * pos_diff4, dim=-1)
    pos_reward4 = torch.exp(-pos_err_scale * pos_err4)

    
    

    reward =  (ball_paddle_contact_buf<1) *  pos_reward *  pos_reward3  * torch.exp(-1 *  paddle_z_dir)  \
        + (solid_ball_paddle_contact_buf>=1) * pos_reward3  * (ball_vel[:,0]>=0) * ((ball_pos[:,2]<=1.4))  * (2 + 1 * pos_reward4  + 1 * (pos_err4<0.01) + 0.0 * ball_vel[:, 0 ] * (pos_err4<0.1) ) * (ball_table_reward_buf <= 1) #+ 1 * (ball_table_reward_buf >= 1) 

    return reward   


# @torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    ball_mytable_contact_buf,
    ball_othertable_contact_buf,
    ball_pos,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0

        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.2, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)
        has_fallen = torch.logical_or(fall_contact, fall_height)

        # has_fallen = fall_height

        
        # 8 is the hand id, need to check
        body_ball_dis = torch.norm(rigid_body_pos - ball_pos[:, None, :], dim=-1)
        body_ball_dis[:, 8] = 1
        # time.sleep()
        # print(body_ball_dis[0])
        ball_contact_body = torch.any(body_ball_dis < 0.2, dim=-1)

        # print(ball_contact_body)
        # time.sleep(0.2)
        # print(ball_othertable_contact_buf[0])
        # time.sleep(0.1)
        # print(ball_mytable_contact_buf[0])
        ball_fail = torch.logical_and(
            torch.logical_or(
                ball_pos[..., 0] < 0, ball_othertable_contact_buf == 0
            ),
            ball_pos[..., 2] < 0.5,
        )
        ball_fail = torch.logical_or(ball_fail, ball_mytable_contact_buf >= 2)
        ball_fail = torch.logical_or(ball_fail, ball_pos[..., 2] > 5)
        ball_fail = torch.logical_or(ball_fail, ball_contact_body)
        # ball_fail = torch.logical_or(ball_fail, ball_pos[...,2]<0.2)

        has_failed = has_fallen
        has_failed = torch.logical_or(has_fallen, ball_fail)

        has_failed *= progress_buf > 2

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )
    
    # if reset[0]:
    #     print(ball_othertable_contact_buf[0])

    return reset, terminated
