import os
from dotenv import load_dotenv  # type: ignore[import]
from dataclasses import dataclass

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


@dataclass
class Config:
    SECRET_KEY: str = os.environ['SECRET_KEY']
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_DATABASE_URI: str = (
        os.getenv("DATABASE_URL") or 'sqlite:///' +
        os.path.join(basedir, 'app.db')
    )
    SEG_MODEL_CKPT: str | None = os.getenv('SEG_MODEL_CKPT')
    MORPH_ADD_SNOW_CKPT: str | None = os.getenv('MORPH_ADD_SNOW_CKPT')
    MORPH_ADD_CLOUDS_CKPT: str | None = os.getenv('MORPH_ADD_CLOUDS_CKPT')
    MORPH_ADD_FOG_CKPT: str | None = os.getenv('MORPH_ADD_FOG_CKPT')
