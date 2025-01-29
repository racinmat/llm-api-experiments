import pandas as pd
from tabulate import tabulate


def print_results(results: list[dict]):
    result_df = pd.DataFrame(results)
    # try:  # this is to handle the case when the terminal is too narrow
    #     columns, rows = os.get_terminal_size()
    # except OSError:
    #     columns, rows = 220, 40
    columns = 250   # I don't want to use whole width, I want to have it readable on GitHub too...
    first_col = 50
    print(tabulate(result_df, headers=["model", "answer"], tablefmt="pretty", showindex=False,
                   maxcolwidths=[first_col, columns - first_col - 2]))
