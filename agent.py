import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import math
from typing import Optional

class RoutePlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_file, route_file, time_limit, money_limit, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        super(RoutePlanningEnv, self).__init__()

        # Load establishments data
        self.establishments = data_file

        # Load route points
        self.route_points = route_file

        self.time_limit = time_limit  # in minutes
        self.money_limit = money_limit  # in dollars

        self.num_establishments = len(self.establishments)
        self.places_visited = np.zeros(self.num_establishments, dtype=np.int32)
        self.current_index = 0  # Start from the first point on the route
        self.time_spent = 0.0
        self.money_spent = 0.0
        self.done = False

        # Track the last activity type and time since last activity
        self.last_activity_type = None
        self.activity_types = list(set(est['activity_type'] for est in self.establishments))
        self.time_since_last_activity = {activity: 99999.0 for activity in self.activity_types}

        # Define action space: indices of establishments plus an action to proceed along the route
        self.action_space = spaces.Discrete(self.num_establishments + 1)

        # Observation space
        obs_size = (1 + 1 + 1 + self.num_establishments + len(self.time_since_last_activity)
                    + 1 + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start  # Initial epsilon value
        self.epsilon_end = epsilon_end  # Minimum epsilon value
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.places_visited = np.zeros(self.num_establishments, dtype=np.int32)
        self.current_index = 0  # Start from the first point on the route
        self.time_spent = 0.0
        self.money_spent = 0.0
        self.done = False
        self.last_activity_type = None
        self.low_reward = False  # Initialize the low reward flag

        for key in self.time_since_last_activity:
            self.time_since_last_activity[key] = 99999.0  # Initialize to a large value

        return self._get_observation(), {}

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        if self.done:
            return self._get_observation(), 0.0, terminated, truncated, info

        # Check constraints before taking action
        if self.time_spent > self.time_limit or self.money_spent > self.money_limit:
            self.done = True
            reward = -100.0  # Penalty for exceeding constraints
            terminated = True
            return self._get_observation(), reward, terminated, truncated, info

        if action == self.num_establishments:
            # Proceed along the route towards the destination
            if self.current_index >= len(self.route_points) - 1:
                # Reached or passed the last point; set as destination
                self.current_index = len(self.route_points) - 1
                self.done = True
                reward = self.calculate_reward(final=True)
                terminated = True
                return self._get_observation(), reward, terminated, truncated, info
            else:
                # Move to the next point on the route
                from_coord = self.route_points[self.current_index]
                to_coord = self.route_points[self.current_index + 1]
                travel_time = estimate_travel_time(from_coord, to_coord)
                travel_cost = 0.0  # Assume no additional cost for moving along the route

                self.time_spent += travel_time
                self.money_spent += travel_cost
                self.current_index += 1

                # Update time since last activity
                for key in self.time_since_last_activity:
                    self.time_since_last_activity[key] += travel_time

                reward = self.calculate_reward(action_taken='move_along_route')
                self.done = False
        else:
            # Visiting an establishment
            if self.places_visited[action] == 1:
                reward = -10.0  # Penalty for revisiting an establishment
                self.done = False
                info['error'] = 'Place already visited'
                return self._get_observation(), reward, terminated, truncated, info
            else:
                establishment = self.establishments[action]
                activity_type = establishment['activity_type']

                # Check if we just had this activity
                if self.time_since_last_activity[activity_type] < 120.0:
                    self.low_reward = True
                else:
                    self.low_reward = False

                if self.current_index >= len(self.route_points):
                    # Cannot visit establishments after reaching destination
                    reward = -100.0  # Penalty for invalid action
                    self.done = True
                    terminated = True
                    info['error'] = 'Cannot visit establishments after reaching destination'
                    return self._get_observation(), reward, terminated, truncated, info
                else:
                    # Simulate visiting the establishment
                    route_coord = self.route_points[self.current_index]
                    establishment_coord = establishment['location']
                    time_to_establishment = estimate_travel_time(route_coord, establishment_coord)
                    time_back_to_route = estimate_travel_time(establishment_coord, route_coord)
                    total_travel_time = time_to_establishment + time_back_to_route
                    total_travel_cost = haversine_distance(route_coord, establishment_coord) * 2 * 0.5  # $0.5 per km

                    # Activity time and cost
                    activity_time = 60.0  # Assume 1 hour spent at the establishment
                    activity_cost = establishment.get('estimated_price', 25.0)

                    total_time_spent = total_travel_time + activity_time
                    total_money_spent = total_travel_cost + activity_cost

                    self.time_spent += total_time_spent
                    self.money_spent += total_money_spent

                    # Update time since last activity
                    for key in self.time_since_last_activity:
                        self.time_since_last_activity[key] += total_time_spent
                    self.time_since_last_activity[activity_type] = 0.0

                    # Mark the establishment as visited
                    self.places_visited[action] = 1

                    # Check constraints after action
                    if self.time_spent > self.time_limit or self.money_spent > self.money_limit:
                        self.done = True
                        reward = -100.0  # Penalty for exceeding constraints
                        terminated = True
                        return self._get_observation(), reward, terminated, truncated, info
                    else:
                        reward = self.calculate_reward(action_taken='visit_establishment')
                        self.done = False

        return self._get_observation(), reward, terminated, truncated, info

    def epsilon_greedy_action(self, q_values):
        """Choose action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Explore: Random action
        else:
            return np.argmax(q_values)  # Exploit: Best action

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)


    def calculate_reward(self, action_taken=None, final=False):
        # Adjust reward scaling
        if final:
            num_establishments_visited = np.sum(self.places_visited)
            if num_establishments_visited == 0:
                return -100.0  # Penalty for not visiting any establishment
            else:
                # Positive reward for visiting establishments and completing route within constraints
                reward = 50.0 + num_establishments_visited * 20.0
                # Penalty for time and money spent
                reward -= (self.time_spent / self.time_limit) * 20.0
                reward -= (self.money_spent / self.money_limit) * 20.0
                return reward
        else:
            if action_taken == 'move_along_route':
                return -2.0  # Small penalty for not visiting an establishment
            elif action_taken == 'visit_establishment':
                last_establishment_indices = np.where(self.places_visited == 1)[0]
                if len(last_establishment_indices) > 0:
                    last_establishment_index = last_establishment_indices[-1]
                    establishment = self.establishments[last_establishment_index]
                    rating = establishment.get('rating', 0.0)
                    if self.low_reward:
                        return rating * 5.0  # Lower reward for repeating activity
                    else:
                        return rating * 10.0  # Higher reward for new activity
                else:
                    return 0.0
            else:
                return 0.0

    def _get_observation(self):
        future_establishments = [
            est for idx, est in enumerate(self.establishments)
            if idx > self.current_index and self.places_visited[idx] == 0
        ]
        if future_establishments:
            max_future_rating = max(est.get('rating', 0.0) for est in future_establishments)
            min_future_price = min(est.get('estimated_price', 0.0) for est in future_establishments)
        else:
            max_future_rating = 0.0
            min_future_price = 0.0

        obs = np.concatenate([
            np.array([self.current_index], dtype=np.float32),
            np.array([self.time_spent], dtype=np.float32),
            np.array([self.money_spent], dtype=np.float32),
            self.places_visited.astype(np.float32),
            np.array(list(self.time_since_last_activity.values()), dtype=np.float32),
            np.array([max_future_rating], dtype=np.float32),
            np.array([min_future_price], dtype=np.float32)
        ])
        return obs

    def render(self, mode='human'):
        print(f"Current Index: {self.current_index}")
        print(f"Time Spent: {self.time_spent}")
        print(f"Money Spent: {self.money_spent}")
        print(f"Places Visited: {self.places_visited}")
        print(f"Time Since Last Activity: {self.time_since_last_activity}")

def haversine_distance(coord1, coord2):
    R = 6371.0  # Earth radius in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance  # in kilometers

def estimate_travel_time(coord1, coord2, speed_kmph=50.0):
    distance = haversine_distance(coord1, coord2)  # in kilometers
    travel_time_hours = distance / speed_kmph
    travel_time_minutes = travel_time_hours * 60
    return travel_time_minutes
