"""
Lvl 4
axis
"""

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

METHODS = ['dqn', 'qrdqn', 'rainbow']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--environment_name', type=str, default='alien')
    parser.add_argument('--result_directory', type=str, default='logs')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    base_dir = Path(args.result_directory)
    for method in METHODS:
        df = pd.read_csv(base_dir / f'{method}.csv')
        df = df[df.environment_name == args.environment_name]

        grouped = df.groupby('frame')
        frames = grouped.frame.mean()
        mean_score = grouped['eval_episode_return'].mean()
        min_score = grouped['eval_episode_return'].min()
        max_score = grouped['eval_episode_return'].max()
        # mean_score = df.group_by('frame')['eval_episode_return'].mean()
        plt.plot(frames, mean_score)
        plt.fill_between(frames, min_score, max_score, alpha=0.2)

    plt.title(args.environment_name.capitalize())
    plt.xlabel('Frames')
    plt.ylabel('Game score')
    plt.show()


if __name__ == '__main__':
    main()


