from flask import Flask
from teshub.server.config import Config
from teshub.server.app.routes import bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(bp)

    return app
