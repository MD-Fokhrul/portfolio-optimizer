import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def plot_portfolio(portfolio, title=None, dims=(15.24, 5.12), holdings_portion=0.75):
    holdings_dims = (dims[0] * holdings_portion, dims[1])
    meta_dims = (dims[0] * (1.0 - holdings_portion), dims[1])
    holdings_plot_img = plot_holdings(portfolio.stock_q, dims=holdings_dims, title=title)
    meta_plot_img = plot_portfolio_meta(portfolio, dims=meta_dims)

    return concat_images([holdings_plot_img, meta_plot_img])


def plot_portfolio_meta(portfolio, dims, y_limit=15000):
    pp = portfolio.purchase_power()
    x = ['cash', 'NW', 'profit']
    y = [portfolio.cash, pp, pp - portfolio.init_cash]
    plt.figure()
    fig = sns.barplot(x=x, y=y).get_figure()
    fig.set_size_inches(dims)
    ax = plt.axes()
    ax.set(ylim=(0, y_limit))
    plt.close()

    return fig_to_img(fig)


def plot_holdings(stock_q, dims, title=None, y_limit=100):
    x = np.array(range(stock_q.shape[0]))
    y = stock_q.squeeze()

    plt.figure()
    fig = sns.barplot(x=x, y=y).get_figure()
    fig.set_size_inches(dims)

    ax = plt.axes()
    if title is not None:
        ax.set_title(title)
    ax.set(ylim=(0, y_limit))

    img = fig_to_img(fig)

    plt.close()

    return img


def concat_images(images):
    total_width = sum([img.width for img in images])
    height = images[0].height

    new_im = Image.new('RGB', (total_width, height))

    x_cursor = 0
    for img in images:
        new_im.paste(img, (x_cursor, 0))
        x_cursor += img.width

    return new_im


def fig_to_img(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer

    return Image.frombytes('RGB', canvas.get_width_height(),
                                canvas.tostring_rgb())

