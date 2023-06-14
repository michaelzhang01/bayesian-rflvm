"""============================================================================
Utility functions for common visualizations.
============================================================================"""

import calendar
from   datetime import datetime
from   jobutils import mkdir

# See here for why to use the 'agg' backend:
# https://stackoverflow.com/a/29172195/1830334
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from   matplotlib.offsetbox import (AnnotationBbox,
                                    OffsetImage)
from   metrics import affine_align
#from   mocap.visualization.sequence import SequenceVisualizer
from   pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from   scipy.special import expit as logistic


# -----------------------------------------------------------------------------
# Base visualizer for data with 2D latent variables.
# -----------------------------------------------------------------------------

class Visualizer:

    def __init__(self, directory, dataset, model_name='rflvm',
                 x_colors=None):
        self.directory  = directory
        self.dataset    = dataset
        if x_colors is not None:
            self.x_colors = x_colors
        elif dataset.has_labels:
            self.x_colors = dataset.labels
        else:
            self.x_colors = 'r'
        self.model_name = model_name
        if dataset.has_true_X:
            self.plot_X(X=dataset.X, suffix='true')

    def plot_X_init(self, X_init):
        self.plot_X(X=X_init, suffix='init')

    def plot_iteration(self, t, Y, F, K, X):
        self.plot_X(t=t, X=X)
        if F is not None:
            self.plot_F(t, F)
        if self.dataset.has_true_K and K is not None:
            self.compare_K(t, K)
        self.compare_Y(t, Y)

    def plot_X(self, X, suffix='', t=-1):
        D      = X.shape[1]
        X_plot = X

        if suffix:
            suffix = f'_{suffix}'

        if D == 2:
            fname = f'{t}_X{suffix}.png'
            X_aligned = affine_align(X_plot, self.dataset.X)
            plt.scatter(X_aligned[:, 0], X_aligned[:, 1], c=self.x_colors)
            self._save(fname)
        elif D == 3:
            # If D == 3, we make both 3D and 2D plots.
            if self.dataset.X is not None:
                fname = f'{t}_X{suffix}_2D.png'
                X_aligned = affine_align(X_plot, self.dataset.X)
                plt.scatter(X_aligned[:, 0], X_aligned[:, 1], c=self.x_colors)
                self._save(fname)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # xx = self.dataset.labels
            ax.plot3D(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c='k', ls='--')
            ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                       c=self.x_colors)
            fname = f'{t}_X{suffix}_3D.png'
            self._save(fname)

        if self.dataset.has_true_X and suffix not in ['true', 'init']:
            self.compare_X_marginals(X=X_plot, t=t)

    def compare_X_marginals(self, X, suffix='', t=-1):
        fname = f'{t}_X{suffix}_marg.png'
        N, D = X.shape
        fig, axes = plt.subplots(2, 1)
        first = True
        titles = ['x coordinate', 'y coordinate']
        X = affine_align(X, self.dataset.X)
        for ax, x_true, x_est, title in zip(axes, self.dataset.X.T, X.T,
                                            titles):
            ax.plot(range(N), x_true, label='true X', color='blue')
            ax.plot(range(N), x_est, label=self.model_name, color='red')
            if first:
                first = False
                ax.legend()
        self._save(fname)

    def plot_F(self, t, F):
        if self.dataset.has_true_F:
            self._compare_F_or_P(self.dataset.F, F, f'{t}_F.png')
        else:
            fname = f'{t}_F.png'
            self._plot_F_or_P(F, fname)

    def plot_P(self, t, F):
        P = logistic(F)
        if self.dataset.has_true_F:
            P_true = logistic(self.dataset.F)
            self._compare_F_or_P(P_true, P, f'{t}_P.png')
        else:
            fname = f'{t}_P.png'
            self._plot_F_or_P(P, fname)

    def _plot_F_or_P(self, F_or_P, fname):
        fig, axes = plt.subplots(5, 1)
        for ax, f_or_p in zip(axes, F_or_P.T[:5]):
            ax.plot(f_or_p)
        self._save(fname)

    def _compare_F_or_P(self, F_or_P_true, F_or_P, fname):
        fig, axes = plt.subplots(5, 1)
        first = True
        for ax, true, inf in zip(axes, F_or_P_true.T[:5], F_or_P.T[:5]):
            ax.plot(true, label='true')
            ax.plot(inf,  label='learned')
            if first:
                first = False
                ax.legend()
        self._save(fname)

    def compare_K(self, t, K, suffix=''):
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_K{suffix}.png'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.dataset.K)
        ax1.set_title('K true')
        ax2.imshow(K)
        ax2.set_title('K approx')
        self._save(fname)

    def compare_Y(self, t, Y, suffix=''):
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_Y{suffix}.png'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.dataset.Y)
        ax1.set_title('Y true')
        ax2.imshow(Y)
        ax2.set_title(self.model_name)
        self._save(fname)

    def plot_hess(self, t, hess):
        fname = f'{t}_hess'
        plt.imshow(hess)
        self._save(fname)

    def _save(self, fname):
        plt.tight_layout()
        plt.savefig(f'{self.directory}/{fname}', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')


# -----------------------------------------------------------------------------
# Utility functions.
# -----------------------------------------------------------------------------

def compare_X_marginals(Xs, titles, X_true, fpath):
    """Compare marginals of multiple Xs. Used when computing baselines.
    """
    D = X_true.shape[1]
    fig, axes = plt.subplots(D, 1)
    fig.set_size_inches(15, 5)

    if X_true is not None:
        Xs     += [X_true]
        titles += ['X true']

    for X, title in zip(Xs, titles):
        X = affine_align(X, X_true)
        first = True
        for ax, x_marg in zip(axes, X.T):
            ax.plot(x_marg, label=title)
            if first:
                first = False
                ax.legend()

    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')

# -----------------------------------------------------------------------------

def get_visualizer(directory, dataset, model_name='rflvm'):
    ds_name = dataset.name

    img_height = 37
    img_width  = 50
    img_cmap   = 'gray'
    return ImageVisualizer(directory=directory,
                           dataset=dataset,
                           model_name=model_name,
                           img_height=img_height,
                           img_width=img_width,
                           img_cmap=img_cmap)


