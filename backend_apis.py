import sys
from sanic import Sanic, response

sys.path.insert(1, "/home/thuge/projects/'VT Hacks'/SideQuests/")

from generate_maps import generate_data
from text_sentiment_comparison import find_best_matches
from train_agent import pipeline

app = Sanic("VTHacks")

@app.route("/search_getaways", methods=["GET"])
async def get_items(request):
    from_destination = request.args.get("from_destination", None)
    to_destination = request.args.get("to_destination", None)
    vibes = request.args.get("vibes", None).split(",")

    # Find similar vibes based on user input
    similar_vibes = find_best_matches(vibes)

    # Generate route points and establishments based on the destinations and vibes
    route_points, establishments = generate_data(from_destination, to_destination, similar_vibes)

    # Run the pipeline with the generated data and convert it into a JSON response
    pipeline_result = pipeline(establishments, route_points)

    # Return the JSON response
    return response.json(pipeline_result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
