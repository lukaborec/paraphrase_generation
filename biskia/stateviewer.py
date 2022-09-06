import numpy as np
import matplotlib.pyplot as plt


class StateViewer(object):

    def __init__(self, experiment=None):
        self.experiment = experiment

    def plot(self, x_p=None, y_p=None, x_t=None, y_t=None, states=None, title=None, fig_name=None, epoch=None):
        """
        :param x_p: optional predicted locations
        :param y_p: optional predicted locations
        :param x_t: optional ground truth locations
        :param y_t: optional ground truth locations
        :param states: optional state locations
        :param title: optional plot title
        :param fig_name: to apply on the figure if experiment is given
        :param epoch: to apply if experiment is given
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if states is not None:
            for idx, state in enumerate(states):
                if np.array_equal(state, np.array([-1, -1, -1])):
                    continue  # Ignore placeholders
                state_x = state[0]
                state_y = state[2]
                ax.scatter(state_x, state_y, marker='s', color="blue", s=5)
                ax.scatter(state_x, state_y, marker='s', color="blue", alpha=0.1, s=480)
                ax.annotate(idx + 1, (state_x, state_y))

        if x_p is not None and y_p is not None:
            ax.scatter(x_p, y_p, marker='o', color="red", s=10)
            ax.scatter(x_p, y_p, marker='o', color="red", alpha=0.3, s=320)

        if x_t is not None and y_t is not None:
            ax.scatter(x_t, y_t, marker='s', color="green")
            ax.scatter(x_t, y_t, marker='s', color="green", alpha=0.3, s=480)  # default: 20

        self.__log_grid_figure(ax, title=title, fig_name=fig_name, epoch=epoch)

    def __log_grid_figure(self, ax, title=None, fig_name=None, epoch=None):
        ax.grid(True)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
        min_value = -1.
        max_value = 1.
        minor_step = .1  # length of a box is 0.1524
        major_step = 1.

        major_ticks = np.arange(min_value, max_value + major_step, step=major_step)
        minor_ticks = np.arange(min_value - minor_step, max_value + minor_step, step=minor_step)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        # display the board
        ax.fill_between(major_ticks, -1, 1, facecolor='gray', alpha=.05)

        # show an minor_step offset if a block is on an edge
        ax.set_xlim([min_value - minor_step, max_value + minor_step])
        ax.set_ylim([min_value - minor_step, max_value + minor_step])

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.9)
        if title:
            plt.title(title)
        if self.experiment:
            self.experiment.log_figure(figure_name=fig_name, figure=plt, step=epoch)
            plt.close()
        else:
            plt.savefig(fig_name)
