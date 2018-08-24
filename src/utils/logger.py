from visdom import Visdom
import numpy as np
import datetime
import pandas as pd

class Printer():

    def __init__(self, batchsize = None, N = None):
        self.batchsize = batchsize
        self.N = N

        self.last=datetime.datetime.now()

    def print(self, stats, iteration):
        print_lst = list()

        if self.N is None:
            print_lst.append('iteration: {}'.format(iteration))
        else:
            print_lst.append('iteration: {}/{}'.format(iteration, self.N))

        dt = (datetime.datetime.now() - self.last).total_seconds()

        print_lst.append('logs/sec: {:.2f}'.format(dt / 1))

        if self.batchsize is not None:
            print_lst.append('samples/sec: {:.2f}'.format(dt / self.batchsize))

        for k, v in zip(stats.keys(), stats.values()):
            print_lst.append('{}: {:.2f}'.format(k, v))

        print('\r' + ', '.join(print_lst), end="")

        self.last = datetime.datetime.now()


class Logger():

    def __init__(self, columns, modes, epoch=0, idx=0):

        self.columns=columns
        self.mode=modes[0]
        self.epoch=epoch
        self.idx = idx
        self.data = pd.DataFrame(columns=["epoch","iteration","mode"]+self.columns)

    def update_epoch(self, epoch=None):
        if epoch is None:
            self.epoch+=1
        else:
            self.epoch=epoch

    def set_mode(self,mode):
        self.mode = mode

    def log(self, stats, iteration):
        self.steps.append(iteration)

        stats["epoch"] = self.epoch
        stats["iteration"] = iteration
        stats["mode"] = self.mode

        row = pd.DataFrame(stats, index=[self.idx])

        self.data = self.data.append(row)

    # def update_visdom_current_epoch(self):
    #
    #     for key in self.record.keys():
    #
    #         if key in self.windows.keys():
    #             win = self.windows[key]
    #         else:
    #             win = None # first log -> new window
    #
    #         opts = dict(
    #             title="current epoch",
    #             showlegend=True,
    #             xlabel='steps',
    #             ylabel=key)
    #
    #         self.windows[key] = self.viz.line(
    #             X=np.array(self.steps),
    #             Y=np.array(self.record[key]),
    #             name=self.prefix,
    #             win=win,
    #             opts=opts
    #         )
    #
    # def update_visdom_per_epoch(self):
    #
    #     for key in self.epoch_record.keys():
    #         if key in self.epochwindows.keys():
    #             win = self.epochwindows[key]
    #         else:
    #             win = None # first log -> new window
    #
    #         opts = dict(
    #             title="all epochs",
    #             showlegend=True,
    #             xlabel='epochs',
    #             ylabel=key)
    #
    #         self.epochwindows[key] = self.viz.line(
    #             X=np.array(self.epochs),
    #             Y=np.array(self.epoch_record[key]),
    #             name=self.prefix,
    #             win=win,
    #             opts=opts
    #         )

