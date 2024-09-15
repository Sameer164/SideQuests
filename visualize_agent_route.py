import json
import folium

# Load the journey data
with open('agent_journey.json', 'r') as f:
    journey_data = json.load(f)

full_path = journey_data['full_path']
visited_establishments = journey_data['visited_establishments']
route_points = journey_data['route_points']

# Create a map centered at the starting point
start_lat, start_lon = full_path[0]
m = folium.Map(location=[start_lat, start_lon], zoom_start=12)

# Plot the predefined route
folium.PolyLine(route_points, color='gray', weight=3, opacity=0.7, tooltip='Predefined Route').add_to(m)

# Plot the agent's full path, including deviations
folium.PolyLine(full_path, color='blue', weight=5, opacity=0.8, tooltip='Agent Path').add_to(m)

# Plot visited establishments with a distinct marker
for est in visited_establishments:
    lat, lon = est['location']
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='red', icon='star', prefix='fa'),
        tooltip=f"{est['name']} ({est['activity_type']})\nRating: {est.get('rating', 'N/A')}\nPrice: ${est.get('estimated_price', 'N/A')}"
    ).add_to(m)

# Add markers for the start and end points
start_coord = route_points[0]
end_coord = route_points[-1]
folium.Marker(
    location=start_coord,
    icon=folium.Icon(color='green', icon='play'),
    tooltip='Start Point'
).add_to(m)
folium.Marker(
    location=end_coord,
    icon=folium.Icon(color='black', icon='stop'),
    tooltip='End Point'
).add_to(m)

# Optionally, plot unvisited establishments with different markers
with open('establishments_data.json', 'r') as f:
    all_establishments = json.load(f)

# Get list of visited place_ids
visited_place_ids = set(est['place_id'] for est in visited_establishments)

# Plot unvisited establishments
for est in all_establishments:
    if est['place_id'] in visited_place_ids:
        print(est['name'])
        # print(est['place_id'])
        lat, lon = est['location']
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            tooltip=f"{est['name']} ({est['activity_type']})\nRating: {est.get('rating', 'N/A')}"
        ).add_to(m)

# Save the map to an HTML file
m.save('agent_route_map.html')
print("Map has been saved to 'agent_route_map.html'")
