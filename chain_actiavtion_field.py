# -*- coding: utf-8 -*-
"""
    INFORMATION TITLE
    -----------------------------------------------------------------
    AUTHOR: Mellen Y.Pu
    DATE: 2023/1/29 下午2:10
    FILE: chain_actiavtion_field.py
    PROJ: EvoMorph
    IDE: PyCharm
    EMAIL: yingmingpu@gmail.com
    ----------------------------------------------------------------- 
                                      ONE DOOR OPENS ALL THE WINDOWS.

    @INTRODUCTION: 
     - 
    @FUNCTIONAL EXPLATION:
     - 
    @LAUNCH:
     - 
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def distance(a, b) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def min_max_norm(X, feature_range=(0, 1)):
    r1, r2 = feature_range  # 将数据归一化到 [r1, r2] 之间
    xmin, xmax = X.min(), X.max() # 得到数据的最大最小值
    X_std = (X - xmin) / (xmax - xmin)     # 标准化到 [0, 1]
    return X_std

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Surface:
    """
        Infinite surface of the growth， initalized by height=0
    """

    def __init__(self, width):
        self.width = width
        self.area = np.zeros((width, width))
        self.grow_comp = np.ones((width, width))
        self.bridge_comp = np.ones((width, width))
        self.every_growth = 1.0
        self.miu = 0
        self.sigma = 16
        self.sigma_inhibt = 64
        self.comp_func = lambda x: 20 * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma ** 2)) / (np.sqrt(2 * np.pi) * self.sigma)
        self.inhabit_func = lambda x: -1 * 10 * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma_inhibt ** 2)) / (np.sqrt(2 * np.pi) * self.sigma_inhibt)
        self.iter = 0
        self.au_active_dist = 50
        self.ligand_limit_dist = 60
        self.au_speed = 4
        self.loc_trend_lst = []
        self.line_length = 30
    def evolute(self):
        """
            Evolution process with total morphology.

            1. current(2') = init(1') * loc_trend(sel) + random() - inhabit(sel)
            2. current(3') = current(2') * loc_trend(sel) + random() - inhabit(sel)
            3. ...

        """
        if self.area[0, 0] == 0:
            curr_state = np.ones((self.width, self.width))
        else:
            curr_state = np.zeros((self.width, self.width))

        loc_trend = np.ones((self.width, self.width))

        self.loc_trend_lst = []
        # FIND TWO RANDOM POINT PAIRS
        while len(self.loc_trend_lst) < 20:
            x0, y0 = np.random.randint(0, self.width), np.random.randint(0, self.width)
            x1, y1 = x0 + np.random.randint(-self.line_length, self.line_length), y0 + np.random.randint(-self.line_length, self.line_length)
            if min([x0, x1, y0, y1]) > 0 and max([x0, x1, y0, y1]) < self.width and distance((x0, x1), (y0, y1)) > self.line_length * 0.5:
                self.loc_trend_lst.append((x0, y0, x1, y1))


        if self.iter < 1:
            # ADD BRIDGES BETWEEN TWO POINTS
            for m, n, p, q in self.loc_trend_lst:
                for i in range(m - self.au_active_dist, m + self.au_active_dist):
                    for j in range(n - self.au_active_dist, n + self.au_active_dist):
                        _distance_ = np.sqrt((i - m) ** 2 + (j - n) ** 2)
                        if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):
                            if _distance_ < self.au_active_dist:
                                self.grow_comp[i, j] += self.comp_func(_distance_ * self.grow_comp[i, j] * 1.2)

                for i in range(p - self.au_active_dist, p + self.au_active_dist):
                    for j in range(n - self.au_active_dist, q + self.au_active_dist):
                        _distance_ = np.sqrt((i - p) ** 2 + (j - q) ** 2)
                        if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):
                            if _distance_ < self.au_active_dist:
                                self.grow_comp[i, j] += self.comp_func(_distance_ * self.grow_comp[i, j] * 1.2)

                dist_2p = distance((p, q), (m, n))
                A = n - q
                B = p - m
                C = p * (q - n) + q * (m - p)
                for i in range(p, m):
                    for j in range(q, n):
                        dist_p2l = np.abs(A * i + B * j + C) / np.sqrt(A ** 2 + B ** 2)
                        loc_trend[i, j] += self.comp_func(1 / dist_p2l)

            random = np.random.randn(self.width ** 2).reshape(self.width, self.width)
            self.grow_comp += curr_state * 0.5 + 0 * random * loc_trend
            self.grow_comp = 0.01 * min_max_norm(self.grow_comp)
        else:
            random = np.random.randn(self.width ** 2).reshape(self.width, self.width)
            self.grow_comp += 0 * curr_state + 0 * random * loc_trend

        # self.grow_comp = sigmoid(self.grow_comp)
        mean_ = self.grow_comp.mean()
        for i in range(len(self.grow_comp)):
            for j in range((len(self.grow_comp))):
                if self.grow_comp[i, j] < mean_ * 1.2:
                    self.grow_comp[i, j ] /= 1 + self.grow_comp[i, j ]
                else:
                    self.grow_comp[i, j] *= 1 + self.grow_comp[i, j ]
        # self.grow_comp = np.exp(self.grow_comp) / sum(np.exp(self.grow_comp))
        # Ligand limits the growth of the mountain by r=C, can escape by P(escape)=f(state)

        # self.area += curr_state + self.bridge_comp
        # self.area = 1 / (1 + np.exp(-1 * self.area))
        # self.area = (self.area - np.mean(self.area)) / (np.max(self.area) - np.min(self.area))
        self.iter += 1
        print(self.grow_comp)

    def view_2d(self, title="", save=True, vrange=(-1, 1)):
        """
            plot of the surface.
        """
        if vrange:
            plt.matshow(self.grow_comp, vmin=vrange[0], vmax=vrange[1])
        else:
            plt.matshow(self.grow_comp)
        # fig, axs = plt.subplots(2, 1)
        # axs[0].matshow(self.area)
        # axs[1].matshow(self.grow_comp)
        plt.title(title)
        plt.colorbar()
        if save:
            os.makedirs('./exp_03/', exist_ok=True)
            plt.savefig(f'./exp_03/{title}')
        plt.show()

    def view_3d(self, title=""):
        """
            plot of the surfce with 3d mode.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.width)
        X, Y = np.meshgrid(x, y)
        Z = self.area
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
        if np.max(Z) < 100:
            ax.set_zlim(0, 100)
        elif np.max(Z) < 1000:
            ax.set_zlim(0, 1000)
        else:
            ax.set_zlim(0, 20000)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.axis('off')
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    surface = Surface(100)
    surface.view_2d("Initalized State", vrange=(0, 1))
    for _ in range(100):
        surface.evolute()
        if _ % 1 == 0:
            surface.view_2d(f"Particle Surface Evolution Step: {_}", vrange=False)


