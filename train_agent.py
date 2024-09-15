import numpy as np
import gymnasium as gym
from agent import RoutePlanningEnv
import folium
from folium.plugins import MarkerCluster
import json

def plot_map(route_points, establishments, places_visited):
    # Create a map centered around the first point in the route
    map_center = route_points[0]
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Add route points as a line
    folium.PolyLine(route_points, color="blue", weight=2.5, opacity=1).add_to(route_map)

    # Add establishments to the map
    marker_cluster = MarkerCluster().add_to(route_map)
    for idx, est in enumerate(establishments):
        lat, lng = est['location']
        if places_visited[idx] == 1:
            # Red circle for visited establishments
            folium.CircleMarker(location=[lat, lng], radius=6, color='red', fill=True, fill_color='red',
                                popup=f"{est['name']} (Visited)").add_to(marker_cluster)
        else:
            # Blue circle for not visited establishments
            folium.CircleMarker(location=[lat, lng], radius=6, color='blue', fill=True, fill_color='blue',
                                popup=f"{est['name']}").add_to(marker_cluster)

    # Save the map to an HTML file
    map_filename = "agent_route_map.html"
    route_map.save(map_filename)
    print(f"Map saved to {map_filename}. You can open this file in a browser to view it.")




# Hyperparameters
num_episodes = 10000
num_eval_episodes = 1
learning_rate = 0.1
discount_factor = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Initialize the environment

def pipeline(establishment_data, route_data):
    env = RoutePlanningEnv(data_file=establishment_data, route_file=route_data, time_limit=600, money_limit=800, 
                        epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)

    q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))


    # Function to get the best action (exploit) based on the trained Q-table
    def exploit_action(observation):
        observation_idx = np.argmax(observation)  # Simplification; handle continuous space better if needed
        return np.argmax(q_table[observation_idx])

    # Training Loop
    def train_model():

        for episode in range(num_episodes):
            observation, _ = env.reset()  # Reset environment and get initial observation
            done = False
            total_reward = 0

            while not done:
                observation_idx = np.argmax(observation)  # Simplified; change if observation space needs more discretization
                # Epsilon-greedy action selection
                action = env.epsilon_greedy_action(q_table[observation_idx])

                # Take action and observe results
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_observation_idx = np.argmax(next_observation)

                # Q-learning update rule
                best_next_action = np.argmax(q_table[next_observation_idx])
                q_table[observation_idx][action] += learning_rate * (reward + discount_factor * q_table[next_observation_idx][best_next_action] - q_table[observation_idx][action])

                observation = next_observation  # Move to next state
                total_reward += reward

                # Check if the episode has terminated
                if terminated or truncated:
                    done = True

            # Decay epsilon after each episode
            env.decay_epsilon()

            # Log progress
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {env.epsilon}")

        print("Training complete.")

    # Evaluation loop
    def evaluate_model():
        observation, _ = env.reset()  # Reset environment and get initial observation
        done = False
        total_reward = 0
        visited = []
        results = []
        while not done:
            # Get the action using exploitation (best known action from Q-table)
            observation_idx = np.argmax(observation)  
            action = exploit_action(observation)

            # Take the action and get the result
            next_observation, reward, terminated, truncated, info = env.step(action)

            if action == env.num_establishments:
                route_lat, route_lng = env.route_points[env.current_index]
                results.append({'latitude': route_lat, 'longitude': route_lng, 'type': 'route'})
            else:
                establishment = env.establishments[action]
                est_lat, est_lng = establishment['location']
                visited.append(action)
                results.append({'latitude': est_lat, 'longitude': est_lng, 'type': 'establishment', 'name': establishment['name']})

            observation = next_observation  # Move to next observation
            total_reward += reward

            # Check if the episode has terminated
            if terminated or truncated:
                done = True
                print(f"Episode finished with reward: {total_reward}")
                if terminated:
                    print("Terminated due to time/money limit or reaching destination.")
                if truncated:
                    print("Truncated episode due to environment constraints.")
        result_json = json.dumps(results, indent = 4)
        return result_json

    # Train the model
    train_model()

    # Evaluate the model
    return evaluate_model()
