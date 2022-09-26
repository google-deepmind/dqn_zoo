from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

METHODS = ['dqn', 'double_q', 'qrdqn', 'addqn']

def pretty_matplotlib_config(fontsize=15):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': fontsize})


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--environment_name', type=str, default='alien')
    parser.add_argument('--result_directory', type=str, default='logs')
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--note', type=str, default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    base_dir = Path(args.result_directory)

    plt.figure(figsize=(10, 8))
    pretty_matplotlib_config(24)
    for method in METHODS:
        df = pd.read_csv(base_dir / f'{method}.csv')
        df = df[df.environment_name == args.environment_name]
        df = df[df.frame < args.num_iterations*1e6]

        grouped = df.groupby('frame')
        frames = grouped.frame.mean() / 1e6
        mean_score = grouped['eval_episode_return'].mean()
        min_score = grouped['eval_episode_return'].min()
        max_score = grouped['eval_episode_return'].max()

        if args.smoothing != 0:
            mean_score = mean_score.rolling(args.smoothing).mean()
            min_score = min_score.rolling(args.smoothing).mean()
            max_score = max_score.rolling(args.smoothing).mean()

        plt.plot(frames, mean_score, label=method)
        plt.fill_between(frames, min_score, max_score, alpha=0.2)

    plt.legend()
    plt.title(args.environment_name.capitalize())
    plt.xlabel('Million frames')
    plt.ylabel('Game score on evaluation')
    # plt.savefig(base_dir / f'{args.environment_name}.png', dpi=120)
    plt.savefig(base_dir / 'figures' / f'{args.environment_name}_{args.note}.pdf', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()


