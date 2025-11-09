# td3_optuna_robot_singlethread.py
import math
import numpy as np
import gymnasium as gym
import sys
from controller import Supervisor
import cv2
import optuna
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn

#gym environment
class RobotGymEnv(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=10_000):
        super().__init__()

        #Use discrete set of goal positions
        self.goal_candidates = [
            np.array([-0.38, -3.6], dtype=np.float32),
            np.array([2.4, -3.4], dtype=np.float32),
            np.array([2.8, -1.97], dtype=np.float32),
            np.array([3.4, -1.04], dtype=np.float32),
            np.array([2.65, -0.55], dtype=np.float32),
            np.array([1.58, -0.96], dtype=np.float32),
            np.array([3.06, 0.55], dtype=np.float32),
            np.array([3.06, 2.34], dtype=np.float32),
            np.array([3.00, 1.39], dtype=np.float32),
            np.array([0.3, 1.3], dtype=np.float32),
            np.array([0.45, 2.22], dtype=np.float32),  
            np.array([2.0, 2.49], dtype=np.float32),    
            np.array([2.0, 3.24], dtype=np.float32),  
            np.array([3.5, 3.3], dtype=np.float32),
            np.array([0.22, -3.45], dtype=np.float32),
            np.array([-1.88, 3.48], dtype=np.float32),
            np.array([-0.77, 2.33], dtype=np.float32),
            np.array([-1.95, 1.56], dtype=np.float32),
            np.array([-3.5, 1.0], dtype=np.float32),
            np.array([-3.5, 3.1], dtype=np.float32),
            np.array([-3.5, -0.72], dtype=np.float32),
            np.array([-2.48, -0.45], dtype=np.float32),
            np.array([-2.27, -2.0], dtype=np.float32),
            np.array([-3.12, -1.8], dtype=np.float32),  
            np.array([-3.47, -2.89], dtype=np.float32),    
            np.array([-2.42, -3.49], dtype=np.float32),  
            np.array([-0.7, -3.59], dtype=np.float32),    
            np.array([-0.6, -1.7], dtype=np.float32),  
        ]   

        #robot observations array
        robot_low = np.array([-1, -1, -1,-1], dtype=np.float32)
        robot_high = np.array([1, 1, 1, 1], dtype=np.float32)

        #lidar observations
        self.lidar_range_values = None
        lidar_points = 18
        lidar_low = np.full(lidar_points, -1, dtype=np.float32)
        lidar_high = np.full(lidar_points, 1, dtype=np.float32)

        #combine robot and lidar observation limits
        obs_low = np.concatenate((robot_low, lidar_low))
        obs_high = np.concatenate((robot_high, lidar_high))

        #bservation space previous actions, distance to goal, angle to goal, lidar
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Action space continuous linear velocity and angular velocity
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.robot = self.getSelf() #define the robot using supervisor

        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='RobotGymEnv-v0', max_episode_steps=max_episode_steps)

        self.__timestep = int(self.getBasicTimeStep())
        self.current_step = 0
        self.episode_score = 0
        self.total_timesteps = 0

        self.collision_occurred = False
        self.previous_action = np.array([0.0, 0.0], dtype=np.float32)


    def normalize(self, value, min_val, max_val):
        # Normalize to [-1, 1]
        return np.clip((2 * (value - min_val) / (max_val - min_val)) - 1, -1.0, 1.0)

    def reset(self, *, seed=4, options=None):
        super().reset(seed=seed)

        self.simulationResetPhysics() # reset simulaiton
        self.simulationReset()
    
        # advance one timestep so devices become available
        super().step(self.__timestep)

        #choose a new goal from the fixed list
        self.goal_pos = self.goal_candidates[np.random.randint(0, len(self.goal_candidates))]

        print(f"New goal position: {self.goal_pos}")

        #initialize devices 
        self.left_motor = self.getDevice("left_wheel_joint")
        self.right_motor = self.getDevice("right_wheel_joint") #robots rotational motors
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        # ensure motors are stopped on reset so episodes don't carry over excessive velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        #Sensors
        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.__timestep)

        #intervals
        super().step(self.__timestep)

        pos = np.array(self.robot.getPosition()[0:2])  #robots position
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy + np.random.normal(0, 0.01, size=2) #randomisation with Gaussian noise


        goal_xy = self.goal_pos # goal position

        dx = pos_xy[0] - goal_xy[0]
        dy = pos_xy[1] - goal_xy[1]
        distance = math.sqrt(dx*dx + dy*dy) #euclidian distance


        self.last_distance = distance
        distance_norm = self.normalize(distance, 0.0, 11.314) #normalise distance

        orient = self.robot.getOrientation() #get robot orientation

        x_axis_world_x = orient[0]  # R00
        x_axis_world_y = orient[3]  # R10

        #angle of the robot X-axis 
        heading = np.array([x_axis_world_x, x_axis_world_y])
        heading_angle = math.atan2(heading[1], heading[0])  # -pi , pi
        heading_angle = heading_angle + np.random.normal(0, 0.01)#randomisation with Gaussian noise
        
        # Goal
        goal_posxy = self.goal_pos
        to_goal = goal_posxy - pos[0:2]
        goal_angle = math.atan2(to_goal[1], to_goal[0])

        # Signed difference -pi to pi
        angle_to_goal = goal_angle - heading_angle
        angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
        angle_to_goal_norm = self.normalize(angle_to_goal, -math.pi, math.pi) # normalise to -1 to 1

        lidar_scan = self.get_lidar_scan() #lidar scan

        obs =  np.array([-1, 0, distance_norm, angle_to_goal_norm]).astype(np.float32)
        obs = np.concatenate((obs, lidar_scan))

        info = {}
        return obs, info
    
    def get_lidar_scan(self):
        ranges = self.lidar.getRangeImage() #get raw scan
        h_resolution = self.lidar.getHorizontalResolution() #get reosltuion

        # Convert to numpy array and clip
        ranges = np.array(ranges[::-1], dtype=np.float32)  # reverse order
        ranges = np.clip(ranges, 0.15, 12.0) #clip to range
        ranges[np.isinf(ranges)] = 0.21

        # Take the front half of the scan
        quarter_points = h_resolution // 4
        front_ranges = ranges[quarter_points : 3*quarter_points]  # front 180°

        # Min pooling over 18 sectors
        target_points = 18
        N = len(front_ranges)
        factor = N / target_points
        downsampled = np.zeros(target_points, dtype=np.float32)

        for i in range(target_points):
            start = int(i * factor)
            end = int((i + 1) * factor)
            downsampled[i] = np.min(front_ranges[start:end]) #take min value

        #store min-pooled values for collision checking
        self.lidar_range_values = ranges
        self.pen_minlidar = np.min(downsampled)
        # Normalize for observations
        lidar_scan = self.normalize(downsampled, 0.15, 12.0)

        return lidar_scan

    
    def show_lidar_scan(self, lidar_scan, step=None):
        img_size = 400
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        scale = img_size // 8  # pixels per meter

        angles = np.linspace(-np.pi/2, np.pi/2, len(lidar_scan))
        for r_norm, a in zip(lidar_scan, angles):
            # denormalize from [-1,1] to 0.15m to 12.0 m
            r = ((r_norm + 1) / 2) * (12.0 - 0.15) + 0.15

            x = int(center[0] + r * scale * np.cos(a))
            y = int(center[1] - r * scale * np.sin(a))
            cv2.line(img, center, (x, y), (255, 255, 255), 1)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        if step is not None:
            cv2.putText(img, f"Step: {step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Lidar Scan", img) #dsiplay scan
        cv2.waitKey(1)  # 1 ms delay for live plotting

    

        
    def step(self, action):
        #convert action o linear and angualr velocity
        v = ((1.0 + action[0]) / 2 )/6.7 # linear speed [0, 0.15]
        omega = action[1]# angular speed [-1, 1]

        R = 0.0325 #get radius of wheel
        L = 0.167 #base diameter

        left_speed = (v - (omega * L / 2)) / R #inverse kinematics
        right_speed = (v + (omega * L / 2)) / R

        self.left_motor.setVelocity(left_speed) #set wheel speeds
        self.right_motor.setVelocity(right_speed)

        super().step(self.__timestep)

        # get bservations
        left_vel = self.left_motor.getVelocity() #get velocities of wheels
        right_vel = self.right_motor.getVelocity()
        pos = self.robot.getPosition() #get robot psition
        pos = pos + np.random.normal(0, 0.01, size=2) #randomisation with Gaussian noise
        orient = self.robot.getOrientation() #get robot orientaiton

        x_axis_world_x = orient[0]  # R00
        x_axis_world_y = orient[3]  # R10

        #heading angle of the robot
        heading = np.array([x_axis_world_x, x_axis_world_y])
        heading_angle = math.atan2(heading[1], heading[0])  #-pi to pi range
        heading_angle = heading_angle + np.random.normal(0, 0.01)#randomisation with Gaussian noise

        # Goal position
        goal_posxy = self.goal_pos
        to_goal = goal_posxy - pos[0:2]
        goal_angle = math.atan2(to_goal[1], to_goal[0])

        #get realtive heading to goal
        angle_to_goal = goal_angle - heading_angle
        angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
        angle_to_goal_norm = self.normalize(angle_to_goal, -math.pi, math.pi)

        # Robot pos in  x and y
        pos_xy = np.array([pos[0], pos[1]])
        pos_xy = pos_xy + np.random.normal(0, 0.01, size=2) #randomisation with Gaussian noise

        #normalized position
        pos_xy_norm = np.zeros(2, dtype=np.float32)
        for i in range(2):
            pos_xy_norm[i] = self.normalize(pos_xy[i], -4.0, 4.0)

        dx = pos_xy[0] - goal_posxy[0]
        dy = pos_xy[1] - goal_posxy[1]
        distance = math.sqrt(dx*dx + dy*dy)

        #normalized distance to goal
        distance_norm = self.normalize(distance, 0.0, 11.314)

        # Lidar
        lidar_scan = self.get_lidar_scan() # get lidar scan
        #uncommet for diaplay lidar
        """if self.current_step % 10 == 0:  # update every 10 steps 
            self.show_lidar_scan(lidar_scan, self.current_step)"""

        #the observation array
        states = np.array([action[0], action[1], distance_norm, angle_to_goal_norm])
        self.state = np.concatenate((states, lidar_scan)) #concatonate obs and lidar scan
        
        
        #minimum lidar distance for collision checking
        min_lidar = np.min(self.lidar_range_values)

        terminated = False
        reward = 0.0
        lidar_reward = 0.0
        distance_reward = 0.0
        omega_reward = 0.0
        
        self.current_step += 1
        self.total_timesteps += 1

        # Check for collisions
        contact_points = self.robot.getContactPoints()
        if contact_points:   
            print("Collision detected:", contact_points)
            reward -= 50.0
            terminated = True
            self.current_step = 0
            # mark that a collision occurred so the next reset will reset the robot to origin
            self.collision_occurred = True

        if not terminated:
            
            if distance < 0.2: #within 20 cm of the goal
                reward += 50.0  # reward for reaching goal
                terminated = True  # end episode 
                self.current_step = 0
                print(f"Episode finished successfully")
                self.solved = True
            elif (abs(pos_xy[0]) > 4 or abs(pos_xy[1]) > 4):
                reward -= 50.0  # penalty for going out of bounds
                terminated = True
                self.current_step = 0
            elif min_lidar < 0.2:
                reward -= 50.0  # penalty for collision
                terminated = True
                self.current_step = 0
                self.collision_occurred = True
                print(f"Episode finished due to obstacle proximity")
            else:
                terminated = False

                distance_reward = (self.last_distance - distance) * 500.0  # reward for reducing distance to goal
                reward += distance_reward

                #obstacle penalty
                lidar_reward = -np.exp(-20.0 * (self.pen_minlidar - 0.2)) / 2
                reward += lidar_reward

                #penalty for high angular velocity
                if self.total_timesteps < 5000:
                    omega_reward = -(abs(omega)) 
                    reward += omega_reward
                else:
                    omega_reward = v - (abs(omega)/6.667) 
                    reward += omega_reward
                    pass
                
                self.last_distance = distance
                self.collision_occurred = False  # reset collision flag if no collision this step


        truncated = self.current_step >= 10_000
        if truncated:
            print(f"Episode truncated")
            self.current_step = 0
            reward -= 5.0  # small penalty for truncation

        self.episode_score += reward
        reward = self.normalize(reward, -50.0, 50.0)  # normalize reward to [-1, 1]

        if self.current_step % 100 == 0: #for debugigng
                   print(f"Action: [{action[0]:.2f}, {action[1]:.2f}] Reward: {reward:.3f} Lidar Reward: {lidar_reward:.3f} Linear_vel:{v:.2f} Distance Rew:{distance_reward:.2f} Omega reward: {omega_reward: .2f}")
                
        self.previous_action = action

        return self.state.astype(np.float32), reward, terminated, truncated, {}



#TD3 hyperparameter sampler
def sample_td3_params(trial: optuna.Trial) -> dict:
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    tau = trial.suggest_float("tau", 0.005, 0.05)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000, 1_000_000])
    policy_delay = trial.suggest_int("policy_delay", 1, 3)
    action_noise_std = trial.suggest_float("action_noise_std", 0.05, 0.3)

    return {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "tau": tau,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "policy_delay": policy_delay,
        "train_freq": 2, #fixed for faster training
        "gradient_steps": 1, #fixed
        "action_noise": NormalActionNoise(mean=np.zeros(2), sigma=np.ones(2)*action_noise_std),
        "policy_kwargs": dict(
            net_arch=[400,300],
            activation_fn=nn.ReLU
        )
    }


#Evaluation callback
class TD3TrialEvalCallback(BaseCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=3, eval_freq=5000):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.is_pruned = False
        self.last_mean_reward = None

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True)
            self.eval_idx += 1
            self.last_mean_reward = mean_reward
            self.trial.report(mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True #prune of no importvement
                return False
        return True

#Objective function
N_TIMESTEPS = 50_000
env = Monitor(RobotGymEnv()) #create only one env cus of webots

def objective(trial: optuna.Trial) -> float:
    env.total_timesteps = 0
    env.current_step = 0
    env.collision_occurred = False
    env.previous_action = np.array([0.0, 0.0], dtype=np.float32)
    env.last_distance = None
    env.lidar_range_values = None
    env.state = None


    td3_params = sample_td3_params(trial)
    
    model = TD3("MlpPolicy", env, verbose=0, **td3_params, device="cpu") #TD# model
    eval_callback = TD3TrialEvalCallback(env, trial, n_eval_episodes=3, eval_freq=5000)
    model.learn(total_timesteps=N_TIMESTEPS, callback=eval_callback)
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    return eval_callback.last_mean_reward

# Run Optuna
if __name__ == "__main__":
    N_TRIALS = 50
    TIMEOUT = 4*60*60  # 2 hours

    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2) 

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner) # create study
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT) #run optimisation 

    print("===== Best Trial =====")
    trial = study.best_trial
    print("Value (mean reward):", trial.value)
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    sys.exit(0)
