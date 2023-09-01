from flask import (Blueprint, render_template)

from app.model.user import get_all_user

routes = Blueprint('home', __name__)


@routes.route('/')
def index():
    users = get_all_user()
    return render_template('home.html', users=users)
