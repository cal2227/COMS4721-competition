import math

import pandas as pd
import matplotlib.pyplot as plt


def plot_frequencies(df):
    def apply_cols(df, func):
        for c in df.columns:
            func(df, c)

    def plot_frequency(df, column):
        plot = df.groupby(column).size().plot(kind='bar')
        fig = plot.get_figure()
        fig.suptitle(column)
        fig.savefig("frequency_plots/{}.png".format(column))

    apply_cols(df, plot_frequency)

def plot_frequencies_digest(df):
    def apply_cols(df, func):
        for (i, c) in enumerate(df.columns):
            func(df, i, c)

    def plot_frequency(df, i, column):
        row = math.floor(i / ncols)
        col = i % ncols
        cell = ax[row, col]
        
        df.groupby(column).size().plot(
          subplots=True,
          ax=cell,
          kind='bar')
        cell.set_xlabel("")

    ncols = 4
    nrows = math.ceil(len(df.columns) / ncols)
    fig, ax = plt.subplots(figsize=(40,40), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    apply_cols(df, plot_frequency)
    fig.savefig("frequency_plots/field_frequencies_digest.png")


if __name__ == '__main__':   
    df = pd.read_csv("data/data.csv")
    plot_frequencies(df)
    plot_frequencies_digest(df)
