import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# seaborn plot stocks over time from dataframe
def plot_stocks(data_df, line_width=0.5, dims=(10.24, 5.12)):
    melted_df = pd.melt(data_df.reset_index(), id_vars='index')

    plt.figure()
    fig = sns.lineplot(x='index', y="value", hue='variable', data=melted_df, lw=line_width).get_figure()
    fig.set_size_inches(dims)
    return fig



