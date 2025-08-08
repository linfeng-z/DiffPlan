import numpy as np
from matplotlib import pyplot as plt


def plot_pano(pano_obs, top_obs=None, title=None, size=(8, 8), reversed_order=True):
    fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=size)

    # > disable unused subplots
    for ax_i in [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]:
        axs[ax_i].axis("off")

    front = axs[0][1]
    back = axs[2][1]
    left = axs[1][0]
    right = axs[1][2]

    front.set_title("Front")
    left.set_title("Left")
    back.set_title("Back")
    right.set_title("Right")

    # > counter-clockwise order
    if reversed_order:
        front.imshow(pano_obs[0])
        left.imshow(pano_obs[-1])
        back.imshow(pano_obs[-2])
        right.imshow(pano_obs[-3])
    else:
        front.imshow(pano_obs[0])
        left.imshow(pano_obs[1])
        back.imshow(pano_obs[2])
        right.imshow(pano_obs[3])

    # > render top-down obs if provided
    if top_obs is not None:
        top = axs[1, 1]
        top.axis("on")
        top.set_title("Top")
        top.imshow(top_obs)

    if title is not None:
        fig.suptitle(title)

    return fig


def get_plt_np_array(fig, ax=None):
    """
    from https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    """

    if ax is not None:
        ax.axis("off")
        # To remove the huge white borders
        ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    return image_from_plot
