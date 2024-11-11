import math
import re
from copy import copy

import gymnasium
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from gymnasium.utils import seeding

from marm_waterworld.marm_waterworld_models import (
    Evaders,
    StaticEvaders,
    Pursuers
)

from reward_machines.reward_machine import RewardMachine

FPS = 15

rm_path_0 = 'marm_waterworld/reward_machines/rm_0.txt'
rm_path_1 = 'marm_waterworld/reward_machines/rm_1.txt'
rm_files = {0: rm_path_0, 1: rm_path_1}

# Number of agents required to capture each food particle;
# n_coop_list[i] means that i-th food particle requires n_coop_list[i] agents to be captured
n_coop_list = [1, 1, 1]

class SARMWaterworldBase:
    def __init__(
            self,
            agent_id,
            reward_machine,
            n_pursuers=1,
            n_evaders=3,
            n_coop=1,
            n_sensors=60,
            sensor_range=1.5,
            radius=0.015,
            pursuer_max_accel=0.5,
            pursuer_speed=0.2,
            evader_speed=0.1,
            food_reward=10.0,
            encounter_reward=0.01,
            thrust_penalty=-0.5,
            local_ratio=1.0,
            speed_features=True,
            max_cycles=2000,
            render_mode=None,
            FPS=FPS,
            reward_shaping=True,
            rm_less=False
    ):
        """Input keyword arguments.

        agent_id: id of the agent for the current single-agent environment
        reward_machine: reward machine for the current agent
        n_pursuers: number of agents
        n_evaders: number of food particles present
        n_coop: number of agents required to capture a food particle
        n_sensors: number of sensors on each agent
        sensor_range: range of the sensor
        radius: radius of the agent
        pursuer_max_accel: maximum acceleration of the agents
        pursuer_speed: maximum speed of the agents
        evader_speed: maximum speed of the food particles
        food_reward: reward for getting a food particle
        encounter_reward: reward for being in the presence of food
        thrust_penalty: scaling factor for the negative reward used to penalize large actions
        local_ratio: proportion of reward allocated locally vs distributed globally among all agents
        speed_features: whether to include entity speed in the state space
        max_cycles: maximum number of timesteps in an episode
        rm_less: whether to include or not RM state in the state observation
        """
        self.pixel_scale = 30 * 25
        self.clock = pygame.time.Clock()
        self.FPS = FPS  # Frames Per Second

        # Setting the agent_id for this environment
        self.agent_id = agent_id

        self.handlers = []

        self.n_coop = n_coop
        self.n_evaders = n_evaders
        self.n_pursuers = n_pursuers
        self.n_sensors = n_sensors

        self.base_radius = radius
        self.sensor_range = sensor_range

        self.pursuer_speed = pursuer_speed * self.pixel_scale
        self.evader_speed = evader_speed * self.pixel_scale
        self.speed_features = speed_features

        self.pursuer_max_accel = pursuer_max_accel

        self.encounter_reward = encounter_reward
        self.food_reward = food_reward
        self.local_ratio = local_ratio
        self.thrust_penalty = thrust_penalty

        self.max_cycles = max_cycles

        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]

        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]
        self.last_rewards = [np.float32(0) for _ in range(self.n_pursuers)]

        self.render_mode = render_mode
        self.screen = None
        self.frames = 0
        self.num_agents = self.n_pursuers

        # Loading the RMs
        self.reward_machines = []
        self.num_rm_states = []
        # Loading the RMs, each agent has its own RM associated
        self.reward_shaping = reward_shaping
        if reward_shaping:
            reward_machine.add_reward_shaping(gamma=0.99, rs_gamma=0.9)
        self.num_rm_states.append(len(reward_machine.get_states()))
        self.reward_machines.append(copy(reward_machine))
        self.num_rms = len(self.reward_machines)
        assert self.num_rms == self.num_agents, "There should be an RM per each agent"
        self.rm_less = rm_less

        # Threshold for the signal simulation
        self.thresh = 0.1

        # self.collisions_dict = {f'pursuer_{i}': [] for i in range(self.n_pursuers)}
        self.collisions = set()

        self.get_spaces()
        self._seed()

    def get_spaces(self):
        """Define the action and observation spaces for all the agents."""
        if self.speed_features:
            obs_dim = 3 * self.n_sensors + 1
        else:
            obs_dim = 2 * self.n_sensors + 1

        self.observation_space = []
        for i in range(self.n_pursuers):
            # Increase the shape size by the length of the one-hot encoding of the RM of pursuer i, if rm_less is False
            obs_shape = obs_dim + self.num_rm_states[i] if not self.rm_less else obs_dim
            self.observation_space.append(spaces.Box(
                low=np.float32(-np.sqrt(2)),
                high=np.float32(np.sqrt(2)),
                shape=(obs_shape,),
                dtype=np.float32,
            ))

        act_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        self.action_space = [act_space for _ in range(self.n_pursuers)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        """Create all moving object instances."""
        self.pursuers = []
        self.evaders = []

        x, y = self._generate_coord()
        self.pursuers.append(
            Pursuers(
                'pursuer_{}'.format(self.agent_id),
                x,
                y,
                self.pursuer_max_accel,
                self.pursuer_speed,
                radius=self.base_radius,
                collision_type=1,
                n_sensors=self.n_sensors,
                sensor_range=self.sensor_range,
                speed_features=self.speed_features,
            )
        )

        for i in range(self.n_evaders):
            if i != 3:
                x, y = self._generate_coord()
                vx, vy = (
                    (2 * self.np_random.random(1) - 1) * self.evader_speed,
                    (2 * self.np_random.random(1) - 1) * self.evader_speed,
                )
            else:
                x, y = 0.5 * self.pixel_scale, 0.5 * self.pixel_scale
                vx, vy = 0.0, 0.0

            max_speed = self.evader_speed
            self.evaders.append(
                Evaders(
                    'evader_{}'.format(i),
                    x,
                    y,
                    vx,
                    vy,
                    radius=2 * self.base_radius,
                    collision_type=i + 1000,
                    max_speed=max_speed,
                    n_coop=n_coop_list[i]
                    )
                )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def convert_coordinates(self, value, option="position"):
        """This function converts coordinates in pymunk into pygame coordinates.

        The coordinate system in pygame is:
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
                        |       |                           │
                        |       |                           │
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↓ y
        The coordinate system in pymunk is:
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↑ y
                        |       |                           │
                        |       |                           │
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
        """
        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])

        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self):
        """Generates a random coordinate for an object.
        """
        # Sample random coordinate (x, y) with x, y ∈ [0, pixel_scale]
        coord = self.np_random.random(2) * self.pixel_scale

        return coord

    def _generate_speed(self, speed):
        """Generates random speed (vx, vy) with vx, vy ∈ [-speed, speed]."""
        _speed = (self.np_random.random(2) - 0.5) * 2 * speed

        return _speed[0], _speed[1]

    def add(self):
        """Add all moving objects to PyMunk space."""
        self.space = pymunk.Space()

        for obj_list in [self.pursuers, self.evaders]:
            for obj in obj_list:
                obj.add(self.space)

    def add_bounding_box(self):
        """Create bounding boxes around the window so that the moving object will not escape the view window.

        The four bounding boxes are aligned in the following way:
        (-100, WINDOWSIZE + 100) ┌────┬────────────────────────────┬────┐ (WINDOWSIZE + 100, WINDOWSIZE + 100)
                                 │xxxx│////////////////////////////│xxxx│
                                 ├────┼────────────────────────────┼────┤
                                 │////│    (WINDOWSIZE, WINDOWSIZE)│////│
                                 │////│                            │////│
                                 │////│(0, 0)                      │////│
                                 ├────┼────────────────────────────┼────┤
                                 │xxxx│////////////////////////////│xxxx│
                    (-100, -100) └────┴────────────────────────────┴────┘ (WINDOWSIZE + 100, -100)
        where "x" represents overlapped regions.
        """
        # Bounding dox edges
        pts = [
            (-100, -100),
            (self.pixel_scale + 100, -100),
            (self.pixel_scale + 100, self.pixel_scale + 100),
            (-100, self.pixel_scale + 100),
        ]

        self.barriers = []

        for i in range(4):
            self.barriers.append(
                pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 100)
            )
            self.barriers[-1].elasticity = 0.999
            self.space.add(self.barriers[-1])

    def draw(self):
        """Draw all moving objects in PyGame."""
        for obj_list in [self.pursuers, self.evaders]:
            for obj in obj_list:
                obj.draw(self.screen, self.convert_coordinates)

    def add_handlers(self):
        # Collision handlers for pursuers v.s. evaders
        for pursuer in self.pursuers:
            for obj in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        pursuer.shape.collision_type, obj.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.pursuer_evader_begin_callback
                self.handlers[-1].separate = self.pursuer_evader_separate_callback

        # Collision handlers for evaders v.s. evaders
        for i in range(self.n_evaders):
            for j in range(i, self.n_evaders):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.evaders[i].shape.collision_type,
                            self.evaders[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for pursuers v.s. pursuers
        for i in range(self.n_pursuers):
            for j in range(i, self.n_pursuers):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.pursuers[i].shape.collision_type,
                            self.pursuers[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

    def reset(self):
        self.add_obj()
        self.frames = 0

        # Add objects to space
        self.add()
        self.add_handlers()
        self.add_bounding_box()

        # Reset the reward machines 
        for rm in self.reward_machines:
            rm.reset()

        # Reset the set of collisions, just in case there were some still "active" before the end of the previous episode (especially the simulated ones)
        self.collisions = set()

        # Get observation: observe_list() returns the observation and reward for each pursuer,
        # so then we only need to consider the first component of each pair for the observations
        obs_list = self.observe_list()
        obs_list = [obs_list[id][0] for id in range(self.n_pursuers)]

        self.last_rewards = [np.float32(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = np.array(obs_list, dtype=np.float32)

        return obs_list

    def step(self, action, is_last):
        action = np.asarray(action) * self.pursuer_max_accel
        action = action.reshape(2)
        thrust = np.linalg.norm(action)
        if thrust > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action * (self.pursuer_max_accel / thrust)

        p = self.pursuers[-1]

        # Clip pursuer speed
        _velocity = np.clip(
            p.body.velocity + action * self.pixel_scale,
            -self.pursuer_speed,
            self.pursuer_speed,
        )

        # Set pursuer speed
        p.reset_velocity(_velocity[0], _velocity[1])

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action ** 2).sum())

        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (
                (accel_penalty / self.n_pursuers)
                * np.ones(self.n_pursuers)
                * (1 - self.local_ratio)
        )

        # Assign the current agent the local portion designated by local_ratio
        # self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            self.space.step(1 / self.FPS)
            # for pursuer_name in self.collisions_dict.keys():
            #     if self.collisions_dict[pursuer_name] != []:
            #         print(pursuer_name, 'collided with', self.collisions_dict[pursuer_name])

            # obs_list contains pairs of (observation, reward) for each pursuer
            obs_list = self.observe_list()
            self.last_obs = [obs_list[-1][0]]

            for id in range(self.n_pursuers):
                p = self.pursuers[id]

                # reward for food caught, encountered
                # self.behavior_rewards[id] = (
                #         self.food_reward * p.shape.food_indicator
                #         + self.encounter_reward * p.shape.food_touched_indicator
                # )

                # reward is based solely on the RM reward
                self.behavior_rewards[id] = obs_list[id][1]

                p.shape.food_indicator = 0

            rewards = np.array(self.behavior_rewards)

            local_reward = rewards
            global_reward = local_reward.mean()

            # Distribute local and global rewards according to local_ratio
            self.last_rewards = local_reward * self.local_ratio + global_reward * (
                    1 - self.local_ratio
            )

            self.frames += 1

            # reset collision dict by setting the value of all pursuers' names to empty list
            # for pursuer_name in self.collisions_dict.keys():
            #     self.collisions_dict[pursuer_name] = []

            # Remove only collisions that include the agent that is training
            #regex = re.compile("^pursuer_{}".format(self.agent_id))
            #self.collisions = {s for s in self.collisions if not regex.match(s)}
            # self.collisions = set()

        return self.observe()

    def observe(self):
        return np.array(self.last_obs[-1], dtype=np.float32)

    def observe_list(self):
        observe_list = []

        for i, pursuer in enumerate(self.pursuers):
            evader_distances = []
            evader_velocities = []

            barrier_distances = pursuer.get_sensor_barrier_readings()

            for evader in self.evaders:
                evader_distance, evader_velocity = pursuer.get_sensor_reading(
                    evader.body.position,
                    evader.radius,
                    evader.body.velocity,
                    self.evader_speed,
                )
                evader_distances.append(evader_distance)
                evader_velocities.append(evader_velocity)

            (
                evader_sensor_distance_vals,
                evader_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                evader_distances,
                velocities=evader_velocities
            )

            if pursuer.shape.food_touched_indicator >= 1:
                food_obs = 1
            else:
                food_obs = 0

            rm_rew = 0

            # Signal simulation for decentralised single-agent training
            if self.agent_id == 0:
                if self.reward_machines[i].current_state == 0 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_1+evader_2')
                elif self.reward_machines[i].current_state == 1 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_1+evader_0')
                elif self.reward_machines[i].current_state == 2 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_1+evader_1') 
                #elif self.reward_machines[i].current_state == 3 and np.random.rand() < self.thresh:
                #    self.collisions.add('pursuer_1+evader_1')
                #elif self.reward_machines[i].current_state == 4 and np.random.rand() < self.thresh:
                #    self.collisions.add('pursuer_1+evader_2')
                #elif self.reward_machines[i].current_state == 3:
                #    self.collisions.add('pursuer_1+evader_1')
            # Simulate the signal of pursuer_0 touching evader_2 only whenever the pursuer_1 touches it
            # elif (self.agent_id == 1 and self.reward_machines[i].current_state == 1 and
            #       'pursuer_1+evader_2' in self.collisions and np.random.rand() < 0.5):

            elif self.agent_id == 1:
                if self.reward_machines[i].current_state == 0 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_0+evader_0')
                elif self.reward_machines[i].current_state == 1 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_0+evader_2')
                elif self.reward_machines[i].current_state == 2 and np.random.rand() < self.thresh:
                    self.collisions.add('pursuer_0+evader_1')
                #elif self.reward_machines[i].current_state == 4 and np.random.rand() < self.thresh:
                #    self.collisions.add('pursuer_0+evader_0')
                #elif self.reward_machines[i].current_state == 2:
                #    self.collisions.add('pursuer_0+evader_1')

            _, rm_rew, rm_done = self.reward_machines[i].step(self.reward_machines[i].current_state,
                                                              "&".join(self.collisions), s_info=None,
                                                              add_rs=self.reward_shaping)

            self.last_dones[i] = rm_done

            # concatenate all observations
            if not self.rm_less:
                if self.speed_features:
                    pursuer_observation = np.concatenate(
                        [
                            barrier_distances,
                            evader_sensor_distance_vals,
                            evader_sensor_velocity_vals,
                            np.array([food_obs]),
                            np.array(self.reward_machines[i].get_current_state_feature(), dtype=np.float32)
                        ], dtype=np.float32)
                else:
                    pursuer_observation = np.concatenate(
                        [
                            barrier_distances,
                            evader_sensor_distance_vals,
                            np.array([food_obs]),
                            np.array(self.reward_machines[i].get_current_state_feature(), dtype=np.float32)
                        ], dtype=np.float32)
            else:
                if self.speed_features:
                    pursuer_observation = np.concatenate(
                        [
                            barrier_distances,
                            evader_sensor_distance_vals,
                            evader_sensor_velocity_vals,
                            np.array([food_obs]),
                        ], dtype=np.float32)
                else:
                    pursuer_observation = np.concatenate(
                        [
                            barrier_distances,
                            evader_sensor_distance_vals,
                            np.array([food_obs]),
                        ], dtype=np.float32)

            observe_list.append((pursuer_observation, rm_rew))

        return observe_list

    def get_sensor_readings(self, positions, velocities=None):
        """Get readings from sensors.

        positions: position readings for all objects by all sensors
        velocities: velocity readings for all objects by all sensors
        """
        distance_vals = np.concatenate(positions, axis=1)

        # Sensor only reads the closest object
        min_idx = np.argmin(distance_vals, axis=1)

        # Normalize sensor readings
        sensor_distance_vals = np.amin(distance_vals, axis=1)

        if velocities is not None:
            velocity_vals = np.concatenate(velocities, axis=1)

            # Get the velocity reading of the closest object
            sensor_velocity_vals = velocity_vals[np.arange(self.n_sensors), min_idx]

            return sensor_distance_vals, sensor_velocity_vals

        return sensor_distance_vals

    def pursuer_evader_begin_callback(self, arbiter, space, data):
        """Called when a collision between a pursuer and an evader occurs.

        The counter of the evader increases by 1, if the counter reaches
        n_coop, then, the pursuer catches the evader and gets a reward.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        # Add one collision to evader
        evader_shape.counter += 1

        # Indicate that food is touched by pursuer
        pursuer_shape.food_touched_indicator += 1
        # print(evader_shape.name, 'is in collision with', pursuer_shape.name)

        # Add collision from the set of current collisions
        self.collisions.add('+'.join([pursuer_shape.name, evader_shape.name]))

        if evader_shape.counter >= self.n_coop:
            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1

        return False

    def pursuer_evader_separate_callback(self, arbiter, space, data):
        """Called when a collision between a pursuer and an evader ends.

        If at this moment there are greater or equal than n_coop pursuers
        that collides with this evader, the evader's position gets reset
        and the pursuers involved will be rewarded.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        # Remove collision from the set of collisions
        self.collisions.remove('+'.join([pursuer_shape.name, evader_shape.name]))

        if evader_shape.counter < evader_shape.n_coop:
            # Remove one collision from evader
            evader_shape.counter -= 1
        else:
            # print(evader_shape.name, 'has been captured')
            # self.collisions_dict[pursuer_shape.name].append(evader_shape.name)
            # this was wrong
            #self.collisions.add('+'.join([pursuer_shape.name, evader_shape.name]))
            evader_shape.counter = 0

            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1
            
            # Do not reset evaders: this was done in the original WaterWorld because of how the
            # reward function was defined, but in our case it doesn't make sense to reset them
            # Reset evader position & velocity; evader_1 needs to stay still in the center of the map
            #if evader_shape.name == 'evader_1':
            #    x, y = 0.5 * self.pixel_scale, 0.5 * self.pixel_scale
            #    vx, vy = 0.0, 0.0
            #else:
            #    x, y = self._generate_coord()
            #    vx, vy = self._generate_speed(evader_shape.max_velocity)            

                # Reset evader's position and velocity if and only if it's not evader_1
            #    evader_shape.reset_position(x, y)
            #    evader_shape.reset_velocity(vx, vy)

        pursuer_shape.food_touched_indicator -= 1

    def return_false_begin_callback(self, arbiter, space, data):
        """Callback function that simply returns False."""
        return False

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale, self.pixel_scale)
                )
                pygame.display.set_caption("Waterworld")
            else:
                self.screen = pygame.Surface((self.pixel_scale, self.pixel_scale))

        self.screen.fill((255, 255, 255))
        self.draw()
        self.clock.tick(self.FPS)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
