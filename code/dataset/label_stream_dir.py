import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import glob
from typing import List, Dict, Any
import argparse
from PIL import Image
from itertools import islice
from matplotlib.widgets import Button
import sys

parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('--input_dir', type=str)
parser.add_argument('--all_unverified', type=bool,
                    action=argparse.BooleanOptionalAction)

args = vars(parser.parse_args())

WEATHER_LABELS = ['cloudy', 'clear']
IMAGES_PER_PLOT = 5
IMAGE_SIZE = (300, 300)
MARKS = ['APPROVED', 'REJECTED', 'FINETUNE']


def plot_img(img_path: str, img_title: str, plt) -> None:
    img = Image.open(img_path)
    img = img.resize(IMAGE_SIZE)

    plt.set_title(img_title)
    plt.set_xticks([])
    plt.set_yticks([])
    plt.imshow(img)


def plot_seg_mask(seg_mask_path: str, plt) -> None:
    mask = np.array(sio.loadmat(seg_mask_path).get('mask'))
    plt.set_xticks([])
    plt.set_yticks([])
    plt.imshow(mask)


def update_plot(dirname: str,
                img_paths: List[str],
                ax: plt.Axes,
                pred_info: Dict[str, Any]) -> None:
    assert(len(img_paths) == len(ax[0]) == len(ax[1]))

    for index, img_path in enumerate(img_paths):
        prob = pred_info[img_path]['prob']
        img_title = (
            f'{WEATHER_LABELS[np.argmax(prob)]} (p={round(np.max(prob), 2)})')

        plot_img(os.path.join(dirname, img_path), img_title, ax[0][index])
        plot_seg_mask(os.path.join(dirname,
                                   pred_info[img_path]['segmentation_path']),
                      ax[1][index])


def split_in_chunks(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def mark_dir(dirname: str, mark: str, exit: bool = True) -> None:
    assert(mark in MARKS)

    mark_split = dirname.split('_')
    target_dirname = f'{dirname}_{mark}'
    if mark_split[-1] in MARKS:
        target_dirname = '_'.join([*mark_split[:-1], mark])

    os.rename(dirname, target_dirname)


def enforce_rel_paths_in_pred_info(json_path: str,
                                   write: bool = False) -> Dict[str, Any]:
    with open(json_path, 'r') as f:
        pred_info = json.load(f)
        abs_paths = list(pred_info.keys())

        for abs_path in abs_paths:
            pred_info[os.path.basename(abs_path)] = pred_info.pop(abs_path)

        for pred in pred_info.values():
            pred['segmentation_path'] = os.path.basename(
                pred['segmentation_path'])

    if write:
        with open(json_path, 'w') as f:
            json.dump(pred_info, f, indent=2)

    return pred_info


def validate_dir(dirname: str) -> None:
    json_path = os.path.join(dirname, 'predictions.json')
    pred_info = enforce_rel_paths_in_pred_info(json_path, write=True)
    img_paths = pred_info.keys()

    fig, ax = plt.subplots(nrows=2, ncols=IMAGES_PER_PLOT)
    plt.subplots_adjust(left=0.1, bottom=0.3)

    chunk_gen = (chunk for
                 chunk in split_in_chunks(img_paths, IMAGES_PER_PLOT))
    chunk = next(chunk_gen, ax)
    if len(chunk) != IMAGES_PER_PLOT:
        return

    update_plot(dirname, chunk, ax, pred_info)

    def next_button_clicked(event):
        chunk = next(chunk_gen, ax)

        if len(chunk) != IMAGES_PER_PLOT:
            return

        update_plot(dirname, chunk, ax, pred_info)
        plt.draw()

    def reject_button_clicked(event):
        mark_dir(dirname, 'REJECTED', True)

        plt.close()

    def approve_button_clicked(event):
        mark_dir(dirname, 'APPROVED', True)

        plt.close()

    def finetune_button_clicked(event):
        mark_dir(dirname, 'FINETUNE', True)

        plt.close()

    next_ax = plt.axes([0.15, 0.1, 0.1, 0.1])
    button_next = Button(next_ax, 'Next')
    button_next.on_clicked(next_button_clicked)

    reject_ax = plt.axes([0.35, 0.1, 0.1, 0.1])
    button_reject = Button(reject_ax, 'Reject')
    button_reject.on_clicked(reject_button_clicked)

    approve_ax = plt.axes([0.55, 0.1, 0.1, 0.1])
    button_approve = Button(approve_ax, 'Approve')
    button_approve.on_clicked(approve_button_clicked)

    finetune_ax = plt.axes([0.75, 0.1, 0.1, 0.1])
    button_finetune = Button(finetune_ax, 'Finetune')
    button_finetune.on_clicked(finetune_button_clicked)

    plt.title(dirname)
    plt.show()


if __name__ == '__main__':
    if args['all_unverified']:
        root_dir = 'Data/WINDY_DATASET'

        marks_regex = '[' + '|'.join(MARKS) + ']'
        # TODO: Update
        dirs = glob.iglob('*', root_dir=root_dir)

        for dir in dirs:
            if dir.split('_')[-1] not in MARKS:
                if len(os.listdir(os.path.join(root_dir, dir))) < 5:
                    continue
                validate_dir(os.path.join(root_dir, dir))
