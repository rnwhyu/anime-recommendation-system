import os
from app import App

flask_instance = App()
app = flask_instance.app

if __name__ == '__main__':
    app.run(debug=os.environ.get('DEBUG', True),
            host=os.environ.get('FLASK_RUN_HOST', '127.0.0.1'),
            port=os.environ.get('FLASK_RUN_PORT', 5000))
