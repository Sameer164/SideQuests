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

# Define constants
# ACTIVITY_TYPES = ['indian_restaurant', 'museum', 'swimming_pool', 'hiking_area', 'night_club', 'restaurant']
PRICE_ESTIMATES = {0: 5.0, 1: 15.0, 2: 30.0, 3: 60.0, 4: 100.0}


def get_directions(origin, destination, mode="driving"):
    """
    Fetches directions between the origin and destination using Google Maps API.
    """
    directions_result = gmaps.directions(origin, destination, mode=mode, departure_time="now")
    
    if directions_result:
        return directions_result[0]['overview_polyline']['points']
    else:
        print("No directions found between the specified locations.")
        return None


def decode_route(overview_polyline):
    """
    Decodes the polyline into route points.
    """
    return polyline.decode(overview_polyline)


def sample_route_points(route_points, sample_distance_meters=5000):
    """
    Samples points along the route based on the specified sample distance in meters.
    """
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


def find_establishments_near_points(sampled_points, activity_types, radius=500, max_places_per_point=3):
    """
    Finds establishments around each sampled point based on activity types.
    """
    establishments = []
    unique_place_ids = set()

    for idx, point in enumerate(sampled_points):
        lat, lng = point

        for activity in activity_types:
            try:
                places_result = gmaps.places_nearby(location=(lat, lng),
                                                    radius=radius,
                                                    type=activity)
                count = 0
                for place in places_result.get('results', []):
                    
                    place_id = place['place_id']
                    if place_id not in unique_place_ids:
                        count += 1
                        price_level = place.get('price_level', 2)
                        estimated_price = PRICE_ESTIMATES.get(price_level, 25.0)
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
                        if count >= max_places_per_point:
                            break

            except googlemaps.exceptions.ApiError as e:
                print(f"API Error at point {idx} ({lat}, {lng}): {e}")
                continue
            except Exception as e:
                print(f"Unexpected error at point {idx} ({lat}, {lng}): {e}")
                continue

            # Sleep to respect API rate limits
            time.sleep(0.1)

    return establishments


def save_to_json(data, filename):
    """
    Saves data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)


def generate_data(origin, destination, vibes ,sample_distance_meters=5000):
    """
    Generate_data function to get directions, decode, sample points, find establishments, and save results.
    """
    # Get directions and decode the polyline
    polyline_points = get_directions(origin, destination)
    if polyline_points is None:
        return

    route_points = decode_route(polyline_points)
    # save_to_json(route_points, 'route_points.json')

    # Sample points along the route
    sampled_points = sample_route_points(route_points, sample_distance_meters)
    
    # Find establishments near the sampled points
    establishments = find_establishments_near_points(sampled_points, vibes)
    print(f"Found {len(establishments)} unique establishments along the route.")

    # Save establishments data to JSON
    # save_to_json(establishments, 'establishments_data.json')
    return route_points, establishments


# if __name__ == "__main__":
#     origin = "1600 Amphitheatre Parkway, Mountain View, CA"
#     destination = "1001 R St, Sacramento, CA"
    
#     generate_data(origin, destination, sample_distance_meters=5000)
