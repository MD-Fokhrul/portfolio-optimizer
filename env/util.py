import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def plot_holdings(stock_q, dims=(10.24, 5.12)):
    x = np.array(range(stock_q.shape[0]))
    y = stock_q.squeeze()

    plt.figure()
    fig = sns.barplot(x=x, y=y).get_figure()
    fig.set_size_inches(dims)

    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    plt.close()

    pil_image = Image.frombytes('RGB', canvas.get_width_height(),
                                canvas.tostring_rgb())

    return pil_image

