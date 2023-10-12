import click
import pandas as pd


@click.command()
@click.option('--input_file', help="Csv file with experiment settings - each columns is a variable with command line args.")
@click.option('--id', type=int, help="Start row (passed to .iloc)")
@click.option('--step', type=int, help="Index step.")
def main(input_file, id, step):
    """
    Iterates over df: df.iloc[id + i * step]
    Returns:
        Prints out rows with variables set to values
    """

    df = pd.read_csv(input_file, index_col=0)

    while True:
        id = id + step
        if id >= len(df):
            break

        row = df.iloc[id]
        assert str.isnumeric(str(row.name))

        res = [f"{k}=\'{v}\'" for k, v in row.iteritems()]
        res.append(f"row_id={row.name}")

        print(' '.join(res))

if __name__ == "__main__":
    main()
