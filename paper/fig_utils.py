"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import string

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = 12

cmap = matplotlib.cm.get_cmap("jet")
colornorm = matplotlib.colors.Normalize(vmin=0, vmax=15)
kp_colors = cmap(colornorm(np.arange(15)))
kp_colors = np.concatenate((kp_colors, kp_colors[[13]]), axis=0)
kp_labels = [
    "eye(back)",
    "eye(bottom)",
    "eye(front)",
    "eye(top)",
    "lowerlip",
    "mouth",
    "nose(bottom)",
    "nose(r)",
    "nose(tip)",
    "nose(top)",
    "nosebridge",
    "paw",
    "whisker(I)",  # "whisker(c1)",
    "whisker(III)",  # "whisker(c2)",
    "whisker(II)",  # "whisker(d1)",
    # "whisker(d2)",
]

kp_labels_old = [
    "eye(back)",
    "eye(bottom)",
    "eye(front)",
    "eye(top)",
    "lowerlip",
    "mouth",
    "nose(bottom)",
    "nose(r)",
    "nose(tip)",
    "nose(top)",
    "nosebridge",
    "paw",
    "whisker(c1)",
    "whisker(d2)",
    "whisker(d1)",
]


viscol = [1.0, 100.0 / 255, 200.0 / 255.0]
smcol = [100.0 / 255, 0, 100.0 / 255.0]

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


def add_apml(ax, xpos, ypos, dx=300, dy=300, tp=30):
    x0, x1, y0, y1 = (
        xpos.min() - dx / 2,
        xpos.min() + dx / 2,
        ypos.max(),
        ypos.max() + dy,
    )
    ax.plot(np.ones(2) * (y0 + dy / 2), [x0, x1], color="k")
    ax.plot([y0, y1], np.ones(2) * (x0 + dx / 2), color="k")
    ax.text(y0 + dy / 2, x0 - tp, "P", ha="center", va="top", fontsize="small")
    ax.text(y0 + dy / 2, x0 + dx + tp, "A", ha="center", va="bottom", fontsize="small")
    ax.text(y0 - tp, x0 + dx / 2, "M", ha="right", va="center", fontsize="small")
    ax.text(y0 + dy + tp, x0 + dx / 2, "L", ha="left", va="center", fontsize="small")


def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il


def get_confidence_interval(data, alpha=0.05):
    """
    Get confidence interval for data
    Parameters
    ----------
    data : list or array
        data to get CI for
    alpha : float
        significance level
    Returns
    -------
    CI : tuple
        confidence interval
    """
    data = data[~np.isnan(data)]  # Remove NaNs
    n = len(data)
    # print(n)
    m = np.mean(data)
    std = np.std(data)
    std_err = std / np.sqrt(n)
    confidence_interval = (m - alpha * std_err, m + alpha * std_err)
    return confidence_interval
