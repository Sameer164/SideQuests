import numpy as np
import json
import folium
from stable_baselines3 import DQN
from agent import RoutePlanningEnv

# Parameters
data_file = 'establishments_data.json'
route_file = 'route_points.json'
time_limit = 600.0  # in minutes
money_limit = 300.0  # in dollars

# Create the environment
env = RoutePlanningEnv(data_file, route_file, time_limit, money_limit)

# Load the trained model
model = DQN.load("route_planning_dqn_google_maps", env=env)

# Reset the environment
obs, info = env.reset()
done = False

# Lists to store the agent's journey
full_path = []  # Includes deviations to establishments
visited_establishments = []

# Start at the origin
if env.current_index >= len(env.route_points):
    current_position = env.route_points[-1]
else:
    current_position = env.route_points[env.current_index]
full_path.append(current_position)

while not done:
    # Use the trained model to predict the action
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if action == env.num_establishments:
        # Proceed along the route
        if env.current_index >= len(env.route_points):
            current_position = env.route_points[-1]
        else:
            current_position = env.route_points[env.current_index - 1]
        full_path.append(current_position)
    else:
        # Visited an establishment
        establishment = env.establishments[action]
        visited_establishments.append(establishment)

        # Record path to the establishment
        establishment_coord = establishment['location']

        # From current route position to establishment
        full_path.append(establishment_coord)

        # Return to the same point on the route
        full_path.append(current_position)

# Save the journey data
with open('agent_journey.json', 'w') as f:
    json.dump({
        'full_path': full_path,
        'visited_establishments': visited_establishments,
        'route_points': env.route_points
    }, f)
