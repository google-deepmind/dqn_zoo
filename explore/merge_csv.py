from pathlib import Path
import pandas as pd


ATTRIBUTES = ['frame', 'environment_name', 'eval_episode_return']
# ENVIRONMENTS = [
#     'bank_heist', 'breakout', 'robotank', 'wizard_of_wor', 'pong'
# ]
ENVIRONMENTS = ['assault', 'breakout', 'pong', 'robotank']

def main():
    base_dir = Path('logs')
    df = pd.DataFrame({
        attribute: [] for attribute in ATTRIBUTES
    })
    for name in ENVIRONMENTS:
        method_df = pd.read_csv(base_dir / f"addqn_0.5_50mln_{name}.csv")
        method_df['environment_name'] = name
        method_df = method_df[ATTRIBUTES]
        df = pd.concat((df, method_df))
        print(len(df))

    df.to_csv(base_dir / 'addqn.csv')



if __name__ == '__main__':
    main()


