import googlemaps
import polyline
import os
import json
import time

# Set your Google Maps API key
API_KEY = '[API KEY REMOVED]'

if not API_KEY:
    raise ValueError("Please set the GOOGLE_MAPS_API_KEY environment variable")

gmaps = googlemaps.Client(key=API_KEY)

# Define origin and destination
origin = "1600 Amphitheatre Parkway, Mountain View, CA"
destination = "1001 R St, Sacramento, CA"

# Get directions
directions_result = gmaps.directions(origin,
                                     destination,
                                     mode="driving",
                                     departure_time="now")

if directions_result:
    overview_polyline = directions_result[0]['overview_polyline']['points']
    route_points = polyline.decode(overview_polyline)
else:
    print("No directions found between the specified locations.")
    exit()

# Save the route points to a JSON file
with open('route_points.json', 'w') as f:
    json.dump(route_points, f)



def sample_route_points(route_points, sample_distance_meters=5000):
    sampled_points = []
    total_distance = 0
    last_point = None

    for point in route_points:
        if last_point is not None:
            distance = gmaps.distance_matrix(origins=[last_point],
                                             destinations=[point],
                                             mode='driving')['rows'][0]['elements'][0]['distance']['value']
            total_distance += distance

            if total_distance >= sample_distance_meters:
                sampled_points.append(point)
                total_distance = 0
        else:
            sampled_points.append(point)

        last_point = point

    return sampled_points

sampled_points = sample_route_points(route_points, sample_distance_meters=5000)

establishments = []
unique_place_ids = set()

# Define activity types
activity_types = ['indian_restaurant', 'museum', 'swimming_pool', 'hiking_area', 'night_club', 'restaurant']

# Price estimates in dollars for each price_level and activity_type
price_estimates = {
        0: 5.0,
        1: 15.0,
        2: 30.0,
        3: 60.0,
        4: 100.0
}

for idx, point in enumerate(sampled_points):
    lat, lng = point

    for activity in activity_types:
        try:
            places_result = gmaps.places_nearby(location=(lat, lng),
                                                radius=500,
                                                type=activity)
            count = 0
            for place in places_result.get('results', []):
                
                place_id = place['place_id']
                if place_id not in unique_place_ids:
                    count += 1
                    price_level = place.get('price_level', 2)
                    estimated_price = price_estimates.get(price_level, 25.0)
                    establishment = {
                        'place_id': place_id,
                        'name': place.get('name'),
                        'location': (place['geometry']['location']['lat'], place['geometry']['location']['lng']),
                        'activity_type': activity,
                        'rating': place.get('rating', 0.0),
                        'user_ratings_total': place.get('user_ratings_total', 0),
                        'price_level': price_level,
                        'estimated_price': estimated_price,
                        'vicinity': place.get('vicinity')
                    }
                    establishments.append(establishment)
                    unique_place_ids.add(place_id)
                    if count > 3:
                        break

        except googlemaps.exceptions.ApiError as e:
            print(f"API Error at point {idx} ({lat}, {lng}): {e}")
            continue
        except Exception as e:
            print(f"Unexpected error at point {idx} ({lat}, {lng}): {e}")
            continue

        # Sleep to respect API rate limits
        time.sleep(0.1)

print(f"Found {len(establishments)} unique establishments along the route.")

# Save establishments to a JSON file
with open('establishments_data.json', 'w') as f:
    json.dump(establishments, f)
