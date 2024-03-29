import math

import pandas as pd
import matplotlib.pyplot as plt


def plot_posterior(df, column):
    n = len(df)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title(column)
    ax.set_ylim(0, 1)

    df[df['label'] == 1].groupby(column).size().divide(n) \
        .sort_values(ascending=False).plot(kind='bar', ax=ax)

    fig.savefig("posterior_plots/{}.png".format(column))

def plot_posteriors(df):
    def apply_cols(df, func):
        for c in df.columns:
            func(df, c)

    def plot_posterior(df, column):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(column)
        ax.set_ylim(0, 1)

        df[df['label'] == 1].groupby(column).size().divide(n) \
            .sort_values(ascending=False).plot(kind='bar', ax=ax)

        fig.savefig("posterior_plots/{}.png".format(column))

    n = len(df)
    apply_cols(df, plot_posterior)

def plot_posteriors_digest(df):
    def apply_cols(df, func):
        for (i, c) in enumerate(df.columns):
            func(df, i, c)

    def plot_posterior(df, i, column):
        row = math.floor(i / ncols)
        col = i % ncols
        cell = ax[row, col]
        
        cell.set_title(column)
        cell.get_xaxis().set_visible(False)
        cell.set_ylim(0, 1)

        df[df['label'] == 1].groupby([column, 'label']).size().divide(n) \
            .sort_values(ascending=False).plot(kind='bar', ax=cell)
        
    n = len(df)
    ncols = 4
    nrows = math.ceil(len(df.columns) / ncols)
    fig, ax = plt.subplots(figsize=(40,40), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    apply_cols(df, plot_posterior)
    fig.savefig("posterior_plots/field_posteriors_digest.png")


if __name__ == '__main__':   
    df = pd.read_csv("data/data.csv")
    plot_posteriors(df)
    plot_posteriors_digest(df)
    # plot_posterior(df, '16')
