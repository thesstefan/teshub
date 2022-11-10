import requests
import argparse 
import urllib.parse
import json
import html_to_json
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import re
import os

import logging 
logging.basicConfig(
    encoding='utf-8', 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

WINDY_API_URL = 'https://api.windy.com/api/webcams/v2/list/category=golf'
EMBED_WINDY_WEBCAM_URL = 'https://webcams.windy.com/webcams/public/embed/player'

DEFAULT_WINDY_API_KEY = 'FU6DIxd8jLNYQkSyZF2ftinZD71C7x4I'
DEFAULT_DATASET_PATH = 'dataset'

parser = argparse.ArgumentParser(
        prog='windy_scraper',
        description='''Scrapes streams from windy.com with parameters 
            following the convention of the Windy API.''')

parser.add_argument('--api_key', type=str, default=DEFAULT_WINDY_API_KEY, help='Windy API Key')
parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH, help='Directory where images are stored')
args = parser.parse_args()


def request_webcam_list(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = {'x-windy-key': args.api_key}

    request = requests.get(WINDY_API_URL, headers=headers)
    request.raise_for_status()

    return json.loads(request.text)['result']['webcams']


def get_webcam_images(webcam_id: int, timeframe: str) -> List[str]:
    embed_webcam_link = '{}/{}/{}'.format(EMBED_WINDY_WEBCAM_URL, webcam_id, timeframe)

    request = requests.get(embed_webcam_link)
    request.raise_for_status()

    soup = BeautifulSoup(request.text, 'lxml')
    pattern = re.compile('(_slideFull = )(.+)((?:\n.+)+)(])')
    script = soup.find('script', text=pattern)
    image_list_str = pattern.search(script.text).group(2).removesuffix(',').replace('\'', '"')

    return json.loads(image_list_str)


def download_images(image_url_list: List[str], dst_dir: str) -> None:
    if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) > 5:
        logging.info(f'Stream dir {dst_dir} already exists and '
                     'contains more than 5 files. Skipping...')

        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for image_url in image_url_list:
        img_data = requests.get(image_url).content
        img_path = os.path.join(dst_dir, os.path.basename(image_url))

        logging.info(f'\t Downloading {img_path}...')

        with open(img_path, 'wb') as handler:
            handler.write(img_data)


def main() -> None:
    args = parser.parse_args()
    webcams = request_webcam_list(args)

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    for webcam in webcams:
        webcam_images = get_webcam_images(webcam['id'], timeframe='year')

        logging.info('Downloading images from stream with id {}'.format(webcam['id']))

        download_images(webcam_images, os.path.join(args.dataset_path, webcam['id']))

if __name__ == '__main__':
    main()

