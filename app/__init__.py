import os
from flask import Flask


class App:

    def __init__(self):
        self.app = Flask(__name__,
                         instance_relative_config=True,
                         static_url_path='')
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev'
        self.app.config.from_mapping(DEBUG=os.environ.get("DEBUG", "False"),
                                     BASE_URL="http://localhost:{}".format(
                                         os.environ.get(
                                             "FLASK_RUN_PORT", 5000)))

        from app.controller import (home, recommender)
        self.app.register_blueprint(home.routes)
        self.app.register_blueprint(recommender.routes)
