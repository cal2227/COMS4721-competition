import math

import pandas as pd
import matplotlib.pyplot as plt


def plot_priors(df):
    def apply_cols(df, func):
        for c in df.columns:
            func(df, c)

    def plot_prior(df, column):
        plot = df.groupby(column).size().divide(n) \
            .sort_values(ascending=False).plot(kind='bar')
        fig = plot.get_figure()
        fig.suptitle(column)
        fig.savefig("prior_plots/{}.png".format(column))

    n = len(df)
    apply_cols(df, plot_prior)

def plot_priors_digest(df):
    def apply_cols(df, func):
        for (i, c) in enumerate(df.columns):
            func(df, i, c)

    def plot_prior(df, i, column):
        row = math.floor(i / ncols)
        col = i % ncols
        cell = ax[row, col]
        
        cell.set_title(column)
        cell.get_xaxis().set_visible(False)

        df.groupby(column).size().divide(n) \
            .sort_values(ascending=False).plot(ax=cell, kind='bar')
        
    n = len(df)
    ncols = 4
    nrows = math.ceil(len(df.columns) / ncols)
    fig, ax = plt.subplots(figsize=(40,40), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    apply_cols(df, plot_prior)
    fig.savefig("prior_plots/field_priors_digest.png")


if __name__ == '__main__':   
    df = pd.read_csv("data/data.csv")
    plot_priors(df)
    plot_priors_digest(df)
