import time
from datetime import timedelta
from flask import (Blueprint, Response, request, json)

from app.recommender import Recommender

routes = Blueprint('recommend', __name__)
recommender = Recommender()


@routes.route('/recommend/<user_id>')
def index(user_id):
    use_model = request.args['use-model'] == 'true'
    user_id = int(user_id)

    start = time.time()
    recommendation, seen, result = recommender.get_anime_recommendation(
        user_id, similar_users_count=10, items=100, use_model=use_model)
    end = time.time()

    process_time = timedelta(seconds=(end - start))

    response_dict = {
        'recommendation': json.loads(recommendation.to_json(orient='records')),
        'seen_anime': json.loads(seen.to_json(orient='records')),
        'leave_one_out_result': result,
        'process_time': str(process_time),
    }

    data = json.dumps(response_dict)

    return Response(data, mimetype='application/json')
