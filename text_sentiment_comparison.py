import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

# Load the Universal Sentence Encoder (USE v4) model from TensorFlow Hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define the Google Places types as a constant
GOOGLE_PLACES_TYPES = [
    'car_dealer', 'car_rental', 'car_repair', 'car_wash', 'electric_vehicle_charging_station', 
    'gas_station', 'parking', 'rest_stop', 'farm', 'art_gallery', 'museum', 'performing_arts_theater', 
    'library', 'preschool', 'primary_school', 'school', 'secondary_school', 'university', 
    'amusement_center', 'amusement_park', 'aquarium', 'banquet_hall', 'bowling_alley', 'casino', 
    'community_center', 'convention_center', 'cultural_center', 'dog_park', 'event_venue', 
    'hiking_area', 'historical_landmark', 'marina', 'movie_rental', 'movie_theater', 'national_park', 
    'night_club', 'park', 'tourist_attraction', 'visitor_center', 'wedding_venue', 'zoo', 'accounting', 
    'atm', 'bank', 'american_restaurant', 'bakery', 'bar', 'barbecue_restaurant', 'brazilian_restaurant', 
    'breakfast_restaurant', 'brunch_restaurant', 'cafe', 'chinese_restaurant', 'coffee_shop', 
    'fast_food_restaurant', 'french_restaurant', 'greek_restaurant', 'hamburger_restaurant', 
    'ice_cream_shop', 'indian_restaurant', 'indonesian_restaurant', 'italian_restaurant', 
    'japanese_restaurant', 'korean_restaurant', 'lebanese_restaurant', 'meal_delivery', 'meal_takeaway', 
    'mediterranean_restaurant', 'mexican_restaurant', 'middle_eastern_restaurant', 'pizza_restaurant', 
    'ramen_restaurant', 'restaurant', 'sandwich_shop', 'seafood_restaurant', 'spanish_restaurant', 
    'steak_house', 'sushi_restaurant', 'thai_restaurant', 'turkish_restaurant', 'vegan_restaurant', 
    'vegetarian_restaurant', 'vietnamese_restaurant', 'administrative_area_level_1', 
    'administrative_area_level_2', 'country', 'locality', 'postal_code', 'school_district', 'city_hall', 
    'courthouse', 'embassy', 'fire_station', 'local_government_office', 'police', 'post_office', 
    'dental_clinic', 'dentist', 'doctor', 'drugstore', 'hospital', 'medical_lab', 'pharmacy', 
    'physiotherapist', 'spa', 'bed_and_breakfast', 'campground', 'camping_cabin', 'cottage', 
    'extended_stay_hotel', 'farmstay', 'guest_house', 'hostel', 'hotel', 'lodging', 'motel', 
    'private_guest_room', 'resort_hotel', 'rv_park', 'church', 'hindu_temple', 'mosque', 'synagogue', 
    'barber_shop', 'beauty_salon', 'cemetery', 'child_care_agency', 'consultant', 'courier_service', 
    'electrician', 'florist', 'funeral_home', 'hair_care', 'hair_salon', 'insurance_agency', 'laundry', 
    'lawyer', 'locksmith', 'moving_company', 'painter', 'plumber', 'real_estate_agency', 
    'roofing_contractor', 'storage', 'tailor', 'telecommunications_service_provider', 'travel_agency', 
    'veterinary_care', 'auto_parts_store', 'bicycle_store', 'book_store', 'cell_phone_store', 
    'clothing_store', 'convenience_store', 'department_store', 'discount_store', 'electronics_store', 
    'furniture_store', 'gift_shop', 'grocery_store', 'hardware_store', 'home_goods_store', 
    'home_improvement_store', 'jewelry_store', 'liquor_store', 'market', 'pet_store', 'shoe_store', 
    'shopping_mall', 'sporting_goods_store', 'store', 'supermarket', 'wholesaler', 'athletic_field', 
    'fitness_center', 'golf_course', 'gym', 'playground', 'ski_resort', 'sports_club', 'sports_complex', 
    'stadium', 'swimming_pool', 'airport', 'bus_station', 'bus_stop', 'ferry_terminal', 'heliport', 
    'light_rail_station', 'park_and_ride'
]

# Function to calculate cosine similarity between vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to rank available categories based on similarity to user input
def rank_activities_use(user_string, available_list, depth):
    """
    Ranks Google Places categories based on similarity to the user-provided input using
    the Universal Sentence Encoder (USE).
    """
    # Encode the user input and available categories
    embeddings = use_model([user_string] + available_list)

    user_embedding = embeddings[0]  # First embedding corresponds to the user input
    available_embeddings = embeddings[1:]  # Remaining embeddings correspond to available list

    # Calculate similarity for each available category
    similarities = []
    for idx, available_embedding in enumerate(available_embeddings):
        similarity_score = cosine_similarity(user_embedding, available_embedding)
        similarities.append((available_list[idx], similarity_score))

    # Sort the available categories by similarity score in descending order
    sorted_activities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Extract and return only the sorted activity names, up to the specified depth
    return [activity for activity, score in sorted_activities][:depth]

# Main function to rank activities based on a list of user inputs
def find_best_matches(vibes):
    sorted_actual_types = []
    
    for event in vibes:
        sorted_actual_types += rank_activities_use(event, GOOGLE_PLACES_TYPES, 3)
    
    return sorted_actual_types