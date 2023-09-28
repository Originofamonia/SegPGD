import pandas as pd


def main():
    file = f'results/effects.csv'
    df = pd.read_csv(file)
    result = df.groupby(['eps', 'alpha'])['effect'].mean().reset_index()
    print(result)


if __name__ == '__main__':
    main()
