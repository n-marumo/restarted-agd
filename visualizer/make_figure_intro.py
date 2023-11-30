import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib
import itertools

os.chdir("./result")
ALPHA = 0.8


sns.set_theme()
sns.set(font_scale=1.4)
cmap = plt.get_cmap("tab10")
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["text.usetex"] = True


def get_problem_paths(problem_name):
    path = pathlib.Path(f"./{problem_name}")
    return [p for p in path.iterdir() if p.is_dir()]


def trim_data(df: pd.DataFrame, timeout, tol_obj):
    idxs = np.where(df["obj"] <= tol_obj)[0]
    if len(idxs):
        df = df[: idxs[0] + 1]
    idxs = np.where(df["elapsed_time"] >= timeout)[0]
    if len(idxs):
        df = df[: idxs[0] + 1]
    return df


def plot_data(problem_name, algs, alg_params, yaxis, timeout, tol_obj, legend=False):
    ylabel = {
        "obj": "Objective function value",
        "gradnorm": "Gradient norm",
    }
    problem_paths = get_problem_paths(problem_name)
    for problem_path in problem_paths:
        if problem_path.name.startswith("_"):
            continue
        # e.g.: problem_path = path to "cs/d200_r10_n50_nnz20_xmax1e-01"

        xmax = 0
        ymin = np.inf
        ymax = 0
        for alg in algs:
            file = problem_path / (alg["filename"] + ".csv")
            if not file.exists():
                continue
            df = pd.read_csv(str(file))
            df = trim_data(df, timeout, tol_obj)
            plt.plot(df["elapsed_time"], df[yaxis], **alg["plot_option"])
            xmax = max(xmax, df["elapsed_time"].values[-1])
            ymin = min(ymin, df[yaxis].values[-1])
            ymax = max(ymax, df[yaxis].values[-1])

        plt.xlabel("Wall clock time [sec]")
        plt.ylabel(ylabel[yaxis])
        plt.yscale("log")
        # plt.ylim(top=dfs["gd"][yaxis].values[int(len(dfs["gd"]) / 200)])
        if xmax > timeout:
            plt.xlim(right=timeout * 1.05)
        if ymin < tol_obj:
            plt.ylim(bottom=tol_obj)
        # plt.xlim([xmin, xmax])
        if legend:
            plt.legend()
        if ymax > 1e3:
            plt.ylim(top=1e3)
        plt.tight_layout(pad=0.1)
        param_str = "_".join([k + str(v) for k, v in alg_params.items()])
        plt.savefig(f"{str(problem_path)}_{param_str}_{yaxis}.pdf")
        # plt.show()
        plt.close()


def make_legend(vertical=False):
    sns.set(font_scale=1.5)
    x = [1, 2]
    y = [1, 2]
    algs = [
        {
            "plot_option": {
                "label": r"\textbf{Proposed}",
                "c": "black",
                "alpha": 1,
                "lw": 3,
            },
        },
        {
            "plot_option": {
                "label": "GD",
                "c": cmap(0),
                "alpha": ALPHA,
                "lw": 2,
            },
        },
        {
            "plot_option": {
                "label": "JNJ2018",
                "c": cmap(1),
                "alpha": ALPHA,
                "lw": 2,
            },
        },
        {
            "plot_option": {
                "label": "LL2022",
                "c": cmap(2),
                "alpha": ALPHA,
                "lw": 2,
            },
        },
    ]

    if vertical:
        ncol = 1
        figsize = (2, 1.5)
    else:
        ncol = len(algs)
        figsize = (7.5, 0.45)

    fig = plt.figure("Line plot")
    legend_fig = plt.figure("Legend plot", figsize=figsize)
    ax = fig.add_subplot(111)
    legend_fig.legend(
        [ax.plot(x, y, **alg["plot_option"])[0] for alg in algs],
        [r"\textbf{Proposed}", "GD", "JNJ2018", "LL2022"],
        loc="center",
        ncol=ncol,
    )
    legend_fig.tight_layout(pad=0.1)
    fig.tight_layout(pad=0.1)
    legend_fig.show()
    fig.show()
    legend_fig.savefig("legend_intro.pdf")


def make_figures_various_params():
    Ls = [1e2, 1e3, 1e4]
    Ms = [1e0, 1e1, 1e2]

    for L, M in itertools.product(Ls, Ms):
        print(L, M)
        algs = [
            {
                "filename": f"ourragd_L_dec0.9_L_init{L}_M_init{M}",
                "plot_option": {
                    "label": r"\textbf{Proposed}",
                    "color": "black",
                    "alpha": 1,
                    "linewidth": 3,
                },
            },
            {
                "filename": f"gd_L_dec0.9_L_init{L}",
                "plot_option": {
                    "label": "GD",
                    "color": cmap(0),
                    "alpha": ALPHA,
                    "linewidth": 2,
                },
            },
            {
                "filename": f"jnj2018_L{L}_rho{M}",
                "plot_option": {
                    "label": "JNJ2018",
                    "color": cmap(1),
                    "alpha": ALPHA,
                    "linewidth": 2,
                },
            },
            {
                "filename": f"ll2022_L{L}_rho{M}",
                "plot_option": {
                    "label": "LL2022",
                    "color": cmap(2),
                    "alpha": ALPHA,
                    "linewidth": 2,
                },
            },
        ]
        plot_data(
            "rosenbrock", algs, {"L": L, "M": M}, "obj", timeout=10, tol_obj=1e-15
        )


make_figures_various_params()
make_legend()
