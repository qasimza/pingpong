import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
from rl_games.common.experience import ExperienceBuffer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim
import learning.common_player as common_player
import learning.common_agent as common_agent 
import learning.hrl_agent as hrl_agent
import learning.ase_agent as ase_agent
import learning.hrl_players as hrl_players
import learning.ase_players as ase_players

import learning.ase_models as ase_models
import learning.ase_network_builder as ase_network_builder

import learning.hrl_models as hrl_models
import learning.hrl_network_builder as hrl_network_builder


from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1340, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        return self.model(x)
    

class MOEPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        with open(os.path.join(os.getcwd(), config['hrl_config']), 'r') as f:
            hrl_config = yaml.load(f, Loader=yaml.SafeLoader)
            hrl_config_params = hrl_config['params']

        self.skill_command_size = 5
        
        self.calculate_diversity = False
        self.calculate_disc = True

        super().__init__(config)

        self._task_size = self.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        
        self._build_llc(llc_config_params, llc_checkpoint)
        self._build_llc_forehand(llc_config_params, 'output/pp_llc_forehand/nn/Humanoid.pth')
        self._build_llc_backhand(llc_config_params, 'output/pp_llc_backhand/nn/Humanoid.pth')
        self._build_llc_forehand_slice(llc_config_params, 'output/pp_llc_forehand_slice/nn/Humanoid.pth')
        self._build_llc_backhand_slice(llc_config_params, 'output/pp_llc_backhand_slice/nn/Humanoid.pth')
        self._build_llc_forehand_smash(llc_config_params, 'output/pp_llc_forehand_smash/nn/Humanoid.pth')


        self._build_hrl_forehand(hrl_config_params, 'output/pp_hlc_forehand/nn/Humanoid.pth')
        self._build_hrl_backhand(hrl_config_params, 'output/pp_hlc_backhand/nn/Humanoid.pth')

        self._build_hrl_forehand_slice(hrl_config_params, 'output/pp_hlc_forehand_slice/nn/Humanoid.pth')
        self._build_hrl_backhand_slice(hrl_config_params, 'output/pp_hlc_backhand_slice/nn/Humanoid.pth')

        self._build_hrl_forehand_smash(hrl_config_params, 'output/pp_hlc_forehand_smash/nn/Humanoid.pth')

        self.weights_list = []       
        
        return

    def env_step(self, actions, obs_dict, forehand_actions=None, forehand_slice_actions=None, backhand_actions=None, backhand_slice_actions=None, forehand_smash_actions = None):

        weights = torch.sigmoid(actions[:,self._latent_dim:])

        actions = self.preprocess_actions(actions[:,:self._latent_dim])

        forehand_actions = self.preprocess_actions(forehand_actions)

        forehand_slice_actions = self.preprocess_actions(forehand_slice_actions)
        backhand_slice_actions = self.preprocess_actions(backhand_slice_actions)
    
        backhand_actions = self.preprocess_actions(backhand_actions)

        forehand_smash_actions = self.preprocess_actions(forehand_smash_actions)
        
        obs = obs_dict['obs']

        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        
        for t in range(self._llc_steps):
            
            llc_actions = self._compute_llc_action(obs, actions, self._llc_agent)

            llc_actions = weights * llc_actions
            
            if len(self.forehand_hit_id)>0:
                llc_forehand_actions = self._compute_llc_action(obs, forehand_actions, self._llc_forehand_agent)
                llc_actions[self.forehand_hit_id] += (1-weights[self.forehand_hit_id]) * llc_forehand_actions[self.forehand_hit_id]
                # llc_actions[self.forehand_hit_id] = llc_forehand_actions[self.forehand_hit_id]
            
            if len(self.forehand_slice_id)>0:
                llc_forehand_slice_actions = self._compute_llc_action(obs, forehand_slice_actions, self._llc_forehand_slice_agent)
                llc_actions[self.forehand_slice_id] += (1-weights[self.forehand_slice_id]) * llc_forehand_slice_actions[self.forehand_slice_id]
                # llc_actions[self.forehand_slice_id] = llc_forehand_slice_actions[self.forehand_slice_id]
                
            if len(self.backhand_hit_id)>0:
                llc_backhand_actions = self._compute_llc_action(obs, backhand_actions, self._llc_backhand_agent)
                llc_actions[self.backhand_hit_id] += (1-weights[self.backhand_hit_id]) * llc_backhand_actions[self.backhand_hit_id]
                # llc_actions[self.backhand_hit_id] = llc_backhand_actions[self.backhand_hit_id]
            
            if len(self.backhand_slice_id)>0:
                llc_backhand_slice_actions = self._compute_llc_action(obs, backhand_slice_actions, self._llc_backhand_slice_agent)
                llc_actions[self.backhand_slice_id] += (1-weights[self.backhand_slice_id]) * llc_backhand_slice_actions[self.backhand_slice_id]
                # llc_actions[self.backhand_slice_id] = llc_backhand_slice_actions[self.backhand_slice_id]
            
            if len(self.forehand_smash_id)>0:
                llc_forehand_smash_actions = self._compute_llc_action(obs, forehand_smash_actions, self._llc_forehand_smash_agent)
                llc_actions[self.forehand_smash_id] += (1-weights[self.forehand_smash_id]) * llc_forehand_smash_actions[self.forehand_smash_id]
                # llc_actions[self.forehand_smash_id] = llc_forehand_smash_actions[self.forehand_smash_id]

            obs, curr_rewards, curr_dones, infos = self.env.step(llc_actions)
            
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']
            
            amp_obs = infos['amp_obs']

            curr_disc_reward = self._calc_disc_reward(amp_obs)
            
            disc_rewards += curr_disc_reward      

                
        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps
        # print(disc_rewards)
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        
    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        self._llc_forehand_agent.is_tensor_obses = self.is_tensor_obses
        self._llc_forehand_slice_agent.is_tensor_obses = self.is_tensor_obses
        self._llc_backhand_agent.is_tensor_obses = self.is_tensor_obses
        self._llc_backhand_slice_agent.is_tensor_obses = self.is_tensor_obses
        self._llc_forehand_smash_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def get_action(self, obs_dict, is_determenistic = False):
        obs = obs_dict['obs']

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        proc_obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : proc_obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())
        clamped_actions = torch.clamp(current_action, -1.0, 1.0)
        
        return clamped_actions
    
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life * 10
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        # for i_ in range(n_games):
        #     if games_played >= n_games:
        #         break
        
        while True:
        
            obs_dict = self.env_reset()
            batch_size = 1
            if len(obs_dict['obs'].size()) > len(self.obs_shape):

                #single
                batch_size = obs_dict['obs'].size()[0]#//2
            self.batch_size = batch_size

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            done_indices = []
            
            for n in range(self.max_steps):

                # if self.env.task.run is False:
                #     break
                obs_dict = self.env_reset(done_indices)


                skill_command = obs_dict['obs'][:,-self.skill_command_size:].clone()
        
                self.forehand_hit_id = (skill_command[:, 0] == 1).nonzero(as_tuple=True)[0]
                self.forehand_slice_id = (skill_command[:, 1] == 1).nonzero(as_tuple=True)[0]
                self.backhand_hit_id = (skill_command[:, 2] == 1).nonzero(as_tuple=True)[0]
                self.backhand_slice_id = (skill_command[:, 3] == 1).nonzero(as_tuple=True)[0]
                self.forehand_smash_id = (skill_command[:, 4] == 1).nonzero(as_tuple=True)[0]
                assert torch.all(skill_command.sum(dim=-1)==1)
                # if has_masks:
                #     masks = self.env.get_action_mask()
                #     action = self.get_masked_action(obs_dict, masks, is_determenistic)
                # else:
                
                action = self.get_action(obs_dict, is_determenistic=True)
                
                hrl_forehand_action = self._hrl_forehand_agent.get_action(obs_dict['obs'][:,:-self.skill_command_size], is_determenistic=True)
                hrl_forehand_slice_action = self._hrl_forehand_slice_agent.get_action(obs_dict['obs'][:,:-self.skill_command_size], is_determenistic=True)
                hrl_backhand_action = self._hrl_backhand_agent.get_action(obs_dict['obs'][:,:-self.skill_command_size], is_determenistic=True)
                hrl_backhand_slice_action = self._hrl_backhand_slice_agent.get_action(obs_dict['obs'][:,:-self.skill_command_size], is_determenistic=True)
                hrl_forehand_slmash_action = self._hrl_forehand_smash_agent.get_action(obs_dict['obs'][:,:-self.skill_command_size], is_determenistic=True)

                # obs_dict, r, done, info = self.env_step(self.env, obs_dict, action)
                obs_dict, r, done, info = self.env_step(action,obs_dict, hrl_forehand_action, hrl_forehand_slice_action, hrl_backhand_action, hrl_backhand_slice_action, hrl_forehand_slmash_action)

                cr += r
                steps += 1

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
        
                done_indices = done_indices[:, 0]

        return 
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        return

    

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim + self.env_info['action_space'].shape[0]
        return

    
    
    def _build_hrl_forehand(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = hrl_network_builder.HRLBuilder()
        network_builder.load(network_params)

        network = hrl_models.ModelHRLContinuous(network_builder)
        hrl_forehand_agent_config = self._build_hrl_agent_config(config_params, network)

        self._hrl_forehand_agent = hrl_players.HRLPlayer(hrl_forehand_agent_config, moe=True)
        self._hrl_forehand_agent.restore(checkpoint_file)
        
        return
    
    def _build_hrl_forehand_slice(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = hrl_network_builder.HRLBuilder()
        network_builder.load(network_params)

        network = hrl_models.ModelHRLContinuous(network_builder)
        hrl_forehand_agent_config = self._build_hrl_agent_config(config_params, network)

        self._hrl_forehand_slice_agent = hrl_players.HRLPlayer(hrl_forehand_agent_config, moe=True)
        self._hrl_forehand_slice_agent.restore(checkpoint_file)
        
        return
    
    def _build_hrl_forehand_smash(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = hrl_network_builder.HRLBuilder()
        network_builder.load(network_params)

        network = hrl_models.ModelHRLContinuous(network_builder)
        hrl_forehand_agent_config = self._build_hrl_agent_config(config_params, network)

        self._hrl_forehand_smash_agent = hrl_players.HRLPlayer(hrl_forehand_agent_config, moe=True)
        self._hrl_forehand_smash_agent.restore(checkpoint_file)

        return

    def _build_hrl_backhand(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = hrl_network_builder.HRLBuilder()
        network_builder.load(network_params)

        network = hrl_models.ModelHRLContinuous(network_builder)
        hrl_forehand_agent_config = self._build_hrl_agent_config(config_params, network)

        self._hrl_backhand_agent = hrl_players.HRLPlayer(hrl_forehand_agent_config, moe=True)
        self._hrl_backhand_agent.restore(checkpoint_file)

        return
    
    def _build_hrl_backhand_slice(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = hrl_network_builder.HRLBuilder()
        network_builder.load(network_params)

        network = hrl_models.ModelHRLContinuous(network_builder)
        hrl_forehand_agent_config = self._build_hrl_agent_config(config_params, network)

        self._hrl_backhand_slice_agent = hrl_players.HRLPlayer(hrl_forehand_agent_config, moe=True)
        self._hrl_backhand_slice_agent.restore(checkpoint_file)

        return
    

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        # self._llc_agent.set_eval()
        return

    def _build_llc_forehand(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_forehand_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_forehand_agent.restore(checkpoint_file)
        print("Loaded Forehand LLC checkpoint from {:s}".format(checkpoint_file))

        return
    
    def _build_llc_forehand_slice(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_forehand_slice_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_forehand_slice_agent.restore(checkpoint_file)
        print("Loaded Forehand LLC checkpoint from {:s}".format(checkpoint_file))
        # self._llc_forehand_slice_agent.set_eval()
        return
    
    def _build_llc_forehand_smash(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_forehand_smash_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_forehand_smash_agent.restore(checkpoint_file)
        print("Loaded Forehand LLC checkpoint from {:s}".format(checkpoint_file))
        
        return
    
    def _build_llc_backhand(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_backhand_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_backhand_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        # self._llc_backhand_agent.set_eval()
        return
    
    def _build_llc_backhand_slice(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_backhand_slice_agent = ase_players.ASEPlayer(llc_agent_config)
        self._llc_backhand_slice_agent.restore(checkpoint_file)
        print("Loaded Forehand LLC checkpoint from {:s}".format(checkpoint_file))
        # self._llc_backhand_slice_agent.set_eval()
        return

    def _build_hrl_agent_config(self, config_params, network):
        env_info = copy.deepcopy(self.env_info)

        config = config_params['config']
        config['hrl_config'] = 'ase/data/cfg/train/rlg/hrl_humanoid.yaml'
        config['llc_checkpoint'] = 'Base'
        obs_space = env_info['observation_space']
        obs_size = obs_space.shape[0]-self.skill_command_size

        env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        env_info['num_envs'] = self.env.task.num_envs

        config['network'] = network
        config['env_info'] = env_info

        return config
    
    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])
        llc_env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        llc_env_info['num_envs'] = self.env.task.num_envs
        config = config_params['config']
        config['network'] = network
        config['env_info'] = llc_env_info

        return config

    def _compute_llc_action(self, obs, actions, llc_agent):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = llc_agent.model.a2c_network.eval_actor(obs=processed_obs, ase_latents=z)
        llc_action = mu
        llc_action = llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs, skill='any'):
        if skill is 'any':
            disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward
    