import sys
from sanic import Sanic, response

sys.path.insert(1, "/home/thuge/projects/'VT Hacks'/SideQuests/")

from generate_maps import generate_data
from text_sentiment_comparison import find_best_matches
from train_agent import pipeline
from sanic_cors import CORS

app = Sanic("VTHacks")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/search_getaways", methods=["GET"])
async def get_items(request):
    print("Called")
    # from_destination = request.args.get("from_destination", None)
    # to_destination = request.args.get("to_destination", None)
    # vibes = request.args.get("vibes", None).split(",")
    # budget = int(request.args.get("budget", 0))
    # time_in_mins = int(request.args.get("time", 0))

    # # Find similar vibes based on user input
    # similar_vibes = find_best_matches(vibes)

    # # Generate route points and establishments based on the destinations and vibes
    # route_points, establishments = generate_data(from_destination, to_destination, similar_vibes)

    # # Run the pipeline with the generated data and convert it into a JSON response
    # pipeline_result = pipeline(establishments, route_points, time_in_mins, budget)
    # print(pipeline_result)
    # # Return the JSON response
    pipeline_result = [
        {
            "latitude": 38.840763,
            "longitude": -77.4422149,
            "type": "establishment",
            "name": "Life Time"
        },
        {
            "latitude": 38.9250027,
            "longitude": -77.02289069999999,
            "type": "establishment",
            "name": "Jirraa Lounge"
        },
        {
            "latitude": 37.1298517,
            "longitude": -80.4089389,
            "type": "establishment",
            "name": "Christiansburg"
        }
    ]
    print(pipeline_result)
    return response.json(pipeline_result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
