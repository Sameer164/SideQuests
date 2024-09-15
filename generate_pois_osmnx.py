import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import sklearn
import json

ox.config(use_cache=True, log_console=True)

google_places_types = [
    # Automotive
    'car_dealer', 'car_rental', 'car_repair', 'car_wash', 'electric_vehicle_charging_station',
    'gas_station', 'parking', 'rest_stop',
    
    # Business
    'farm',
    
    # Culture
    'art_gallery', 'museum', 'performing_arts_theater',
    
    # Education
    'library', 'preschool', 'primary_school', 'school', 'secondary_school', 'university',
    
    # Entertainment and Recreation
    'amusement_center', 'amusement_park', 'aquarium', 'banquet_hall', 'bowling_alley', 'casino',
    'community_center', 'convention_center', 'cultural_center', 'dog_park', 'event_venue',
    'hiking_area', 'historical_landmark', 'marina', 'movie_rental', 'movie_theater', 'national_park',
    'night_club', 'park', 'tourist_attraction', 'visitor_center', 'wedding_venue', 'zoo',
    
    # Finance
    'accounting', 'atm', 'bank',
    
    # Food and Drink
    'american_restaurant', 'bakery', 'bar', 'barbecue_restaurant', 'brazilian_restaurant',
    'breakfast_restaurant', 'brunch_restaurant', 'cafe', 'chinese_restaurant', 'coffee_shop',
    'fast_food_restaurant', 'french_restaurant', 'greek_restaurant', 'hamburger_restaurant',
    'ice_cream_shop', 'indian_restaurant', 'indonesian_restaurant', 'italian_restaurant',
    'japanese_restaurant', 'korean_restaurant', 'lebanese_restaurant', 'meal_delivery',
    'meal_takeaway', 'mediterranean_restaurant', 'mexican_restaurant', 'middle_eastern_restaurant',
    'pizza_restaurant', 'ramen_restaurant', 'restaurant', 'sandwich_shop', 'seafood_restaurant',
    'spanish_restaurant', 'steak_house', 'sushi_restaurant', 'thai_restaurant', 'turkish_restaurant',
    'vegan_restaurant', 'vegetarian_restaurant', 'vietnamese_restaurant',
    
    # Geographical Areas
    'administrative_area_level_1', 'administrative_area_level_2', 'country', 'locality',
    'postal_code', 'school_district',
    
    # Government
    'city_hall', 'courthouse', 'embassy', 'fire_station', 'local_government_office', 'police',
    'post_office',
    
    # Health and Wellness
    'dental_clinic', 'dentist', 'doctor', 'drugstore', 'hospital', 'medical_lab', 'pharmacy',
    'physiotherapist', 'spa',
    
    # Lodging
    'bed_and_breakfast', 'campground', 'camping_cabin', 'cottage', 'extended_stay_hotel',
    'farmstay', 'guest_house', 'hostel', 'hotel', 'lodging', 'motel', 'private_guest_room',
    'resort_hotel', 'rv_park',
    
    # Places of Worship
    'church', 'hindu_temple', 'mosque', 'synagogue',
    
    # Services
    'barber_shop', 'beauty_salon', 'cemetery', 'child_care_agency', 'consultant', 'courier_service',
    'electrician', 'florist', 'funeral_home', 'hair_care', 'hair_salon', 'insurance_agency',
    'laundry', 'lawyer', 'locksmith', 'moving_company', 'painter', 'plumber', 'real_estate_agency',
    'roofing_contractor', 'storage', 'tailor', 'telecommunications_service_provider', 'travel_agency',
    'veterinary_care',
    
    # Shopping
    'auto_parts_store', 'bicycle_store', 'book_store', 'cell_phone_store', 'clothing_store',
    'convenience_store', 'department_store', 'discount_store', 'electronics_store', 'furniture_store',
    'gift_shop', 'grocery_store', 'hardware_store', 'home_goods_store', 'home_improvement_store',
    'jewelry_store', 'liquor_store', 'market', 'pet_store', 'shoe_store', 'shopping_mall',
    'sporting_goods_store', 'store', 'supermarket', 'wholesaler',
    
    # Sports
    'athletic_field', 'fitness_center', 'golf_course', 'gym', 'playground', 'ski_resort',
    'sports_club', 'sports_complex', 'stadium', 'swimming_pool',
    
    # Transportation
    'airport', 'bus_station', 'bus_stop', 'ferry_terminal', 'heliport', 'light_rail_station',
    'park_and_ride'
]

def get_route_graph(from_location, to_location):
    # Create a graph from OSM data within a bounding box that covers both locations
    north = max(from_location[0], to_location[0]) + 0.1
    south = min(from_location[0], to_location[0]) - 0.1
    east = max(from_location[1], to_location[1]) + 0.1
    west = min(from_location[1], to_location[1]) - 0.1

    G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    orig_node = ox.nearest_nodes(G, X=from_location[1], Y=from_location[0])
    dest_node = ox.nearest_nodes(G, X=to_location[1], Y=to_location[0])

    # Compute the shortest path
    route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    return G, route


def get_pois_along_route(G, route, tags):
    # Split the route into smaller segments
    segment_length = 5000  # Approximate length of each segment in meters
    nodes = route
    segments = []
    current_segment = []
    current_length = 0.0

    for i in range(len(nodes) - 1):
        u = nodes[i]
        v = nodes[i + 1]
        data = G.get_edge_data(u, v)
        length = data[0].get('length', 0.0)
        current_length += length
        current_segment.append(u)
        if current_length >= segment_length:
            current_segment.append(v)
            segments.append(current_segment)
            current_segment = []
            current_length = 0.0

    # Add the last segment
    if current_segment:
        current_segment.append(nodes[-1])
        segments.append(current_segment)

    # For each segment, query POIs
    all_pois = []
    for segment in segments:
        edge_geometries = []
        for u, v in zip(segment[:-1], segment[1:]):
            data = G.get_edge_data(u, v)
            geometry = data[0].get('geometry', None)
            if geometry:
                edge_geometries.append(geometry)
            else:
                point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
                point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
                edge_geometries.append(LineString([point_u, point_v]))
        segment_line = LineString([point for geom in edge_geometries for point in geom.coords])
        segment_buffer = segment_line.buffer(0.005)  # 500 meters
        try:
            pois = ox.geometries_from_polygon(segment_buffer, tags)
            all_pois.append(pois)
        except Exception as e:
            print(f"Failed to get POIs for a segment: {e}")
            continue

    # Combine all POIs into a single GeoDataFrame
    if all_pois:
        pois_combined = pd.concat(all_pois, ignore_index=True)
        return pois_combined
    else:
        return pd.DataFrame()


def main():
    # Define multiple routes for generalization
    routes = [
        # {'from': (37.7749, -122.4194), 'to': (34.0522, -118.2437)},  # San Francisco to Los Angeles
        # {'from': (40.7128, -74.0060), 'to': (38.9072, -77.0369)},    # New York to Washington D.C.
        {'from': (47.6062, -122.3321), 'to': (45.5152, -122.6784)},  # Seattle to Portland
        # {'from': (34.0522, -118.2437), 'to': (36.1699, -115.1398)},  # Los Angeles to Las Vegas
        # {'from': (25.7617, -80.1918), 'to': (28.5383, -81.3792)},    # Miami to Orlando
        # Add more routes as needed
    ]

    # Define the types of POIs to extract
    tags = {'amenity': google_places_types}

    data = []

    for route_info in routes:
        from_location = route_info['from']
        to_location = route_info['to']
        print(f"Processing route from {from_location} to {to_location}")

        try:
            G, route = get_route_graph(from_location, to_location)
        except Exception as e:
            print(f"Failed to get route: {e}")
            continue

        try:
            pois = get_pois_along_route(G, route, tags)
        except Exception as e:
            print(f"Failed to get POIs: {e}")
            continue

        # Convert POIs to DataFrame and reset index
        pois = pois.reset_index()

        # If no POIs found, skip this route
        if pois.empty:
            print("No POIs found along the route.")
            continue

        # Assign estimated ratings and prices
        pois['rating'] = np.random.uniform(3.0, 5.0, size=len(pois))
        pois['price_level'] = np.random.randint(1, 4, size=len(pois))
        pois['activity_time'] = 60.0  # Assume 60 minutes for activity

        # Build a list of waypoints
        waypoints = []
        for idx, row in pois.iterrows():
            waypoint = {
                'name': row.get('name', 'Unnamed'),
                'location': (row.geometry.y, row.geometry.x),
                'amenity': row.get('amenity', 'unknown'),
                'rating': row['rating'],
                'price_level': row['price_level'],
                'activity_time': row['activity_time']
            }
            waypoints.append(waypoint)

        # Precompute travel times between all pairs of waypoints and from/to locations
        # Since we don't have actual travel times, we can estimate based on Euclidean distance
        all_locations = [from_location] + [wp['location'] for wp in waypoints] + [to_location]
        travel_times = {}
        for i, origin in enumerate(all_locations):
            for j, destination in enumerate(all_locations):
                if i != j:
                    key = f"{i}-{j}"
                    distance = ox.distance.great_circle_vec(origin[0], origin[1], destination[0], destination[1])
                    # Assume average speed of 50 km/h to calculate travel time
                    travel_time = (distance / 1000) / 50 * 60  # Convert to minutes
                    travel_times[key] = {'travel_time': travel_time, 'distance': distance / 1000}  # Distance in km

        route_data = {
            'from_location': from_location,
            'to_location': to_location,
            'waypoints': waypoints,
            'travel_times': travel_times
        }
        data.append(route_data)

    # Save data to a JSON file
    with open('precomputed_data_osmnx.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()
