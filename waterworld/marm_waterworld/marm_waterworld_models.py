import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pygame import Color

pursuer_colors = {'pursuer_0': Color(10, 135, 84), 'pursuer_1': Color(16, 152, 247)}

evader_colors = {'evader_0': Color(88, 81, 114),
                 'evader_1': Color(40, 175, 176),
                 'evader_2': Color(234, 150, 75),
                 'evader_3': Color(107, 94, 98)}


class MovingObject:
    def __init__(self, name, x, y, pixel_scale=750, radius=0.015, color=Color(0, 0, 0)):

        self.name = name

        self.pixel_scale = 30 * 25
        self.body = pymunk.Body()
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity

        self.radius = radius * pixel_scale

        self.color = color

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy


class Evaders(MovingObject):
    def __init__(self, name, x, y, vx, vy, radius=0.03, collision_type=2, max_speed=100, n_coop=1):
        super().__init__(name, x, y, radius=radius, color=evader_colors[name])
        self.body.velocity = vx, vy

        self.color = evader_colors[name]
        self.shape.name = name
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.n_coop = n_coop
        self.shape.max_velocity = max_speed
        self.shape.density = 0.01


class StaticEvaders:
    def __init__(self, name, x=375, y=375, vx=0, vy=0, radius=0.03, collision_type=2, max_speed=0, n_coop=1):
        self.name = name

        self.pixel_scale = 30 * 25
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = x, y
        
        self.initial_x = x
        self.initial_y = y
        self.initial_vx = vx
        self.initial_vy = vy

        self.shape = pymunk.Circle(self.body, self.pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity

        self.radius = radius * self.pixel_scale

        self.body.velocity = vx, vy

        self.color = evader_colors[name]
        self.shape.name = name
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.n_coop = n_coop
        self.shape.max_velocity = max_speed
        self.shape.density = 0.01

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        self.body.position = self.initial_x, self.initial_y

    def reset_velocity(self, vx, vy):
        self.body.velocity = self.initial_vx, self.initial_vy


class Pursuers(MovingObject):
    def __init__(
        self,
        name,
        x,
        y,
        max_accel,
        pursuer_speed,
        radius=0.015,
        n_sensors=30,
        sensor_range=0.2,
        collision_type=1,
        speed_features=True,
    ):
        super().__init__(name, x, y, radius=radius)

        self.color = pursuer_colors[name]
        self.shape.name = name
        self.shape.collision_type = collision_type
        self.sensor_color = (0, 0, 0)
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range * self.pixel_scale
        self.max_accel = max_accel
        self.max_speed = pursuer_speed
        self.body.velocity = 0.0, 0.0

        self.shape.food_indicator = 0  # 1 if food caught at this step, 0 otherwise
        self.shape.food_touched_indicator = (
            0  # 1 if food touched as this step, 0 otherwise
        )
        self.shape.poison_indicator = 0  # 1 if poisoned this step, 0 otherwise

        # Generate self.n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors + 1)[:-1]

        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors
        self.shape.custom_value = 1

        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 2
        if speed_features:
            self._sensor_obscoord += 1

        self.sensor_obs_coord = self.n_sensors * self._sensor_obscoord
        self.obs_dim = (
            self.sensor_obs_coord + 1
        )  # +1 for is_colliding_evader

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-2 * np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.Box(
            low=np.float32(-self.max_accel),
            high=np.float32(self.max_accel),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def position(self):
        assert self.body.position is not None
        return np.array([self.body.position[0], self.body.position[1]])

    @property
    def velocity(self):
        assert self.body.velocity is not None
        return np.array([self.body.velocity[0], self.body.velocity[1]])

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def draw(self, display, convert_coordinates):
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self.sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_barrier_readings(self):
        """Get the distance to the barrier.

        See https://github.com/BolunDai0216/WaterworldRevamp for
        a detailed explanation.
        """
        # Get the endpoint position of each sensor
        sensor_vectors = self._sensors * self.sensor_range
        position_vec = np.array([self.body.position.x, self.body.position.y])
        sensor_endpoints = position_vec + sensor_vectors

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, 0.0, self.pixel_scale)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - position_vec

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio will limit the end of the vector to the barriers
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=np.abs(sensor_vectors) > 1e-8,
        )

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to 1.0
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = 1.0

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0

        return sensor_values[0, :]

    def get_sensor_reading(
        self, object_coord, object_radius, object_velocity, object_max_velocity
    ):
        """Get distance and velocity to another object (Obstacle, Pursuer, Evader, Poison)."""
        # Get location and velocity of pursuer
        self.center = self.body.position
        _velocity = self.body.velocity

        # Get distance of object in local frame as a 2x1 numpy array
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # Get relative velocity as a 2x1 numpy array
        relative_speed = np.array(
            [
                [object_velocity[0] - _velocity[0]],
                [object_velocity[1] - _velocity[1]],
            ]
        )

        # Project distance to sensor vectors
        sensor_distances = self._sensors @ distance_vec

        # Project velocity vector to sensor vectors
        sensor_velocities = (
            self._sensors @ relative_speed / (object_max_velocity + self.max_speed)
        )

        # if np.any(sensor_velocities < -2 * np.sqrt(2)) or np.any(
        #     sensor_velocities > 2 * np.sqrt(2)
        # ):
        #     set_trace()

        # Check for valid detection criterions
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self.sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # Set not sensed sensor readings of position to sensor range
        sensor_distances = np.clip(sensor_distances / self.sensor_range, 0, 1)
        sensor_distances[not_sensed_idx] = 1.0

        # Set not sensed sensor readings of velocity to zero
        sensor_velocities[not_sensed_idx] = 0.0

        return sensor_distances, sensor_velocities
