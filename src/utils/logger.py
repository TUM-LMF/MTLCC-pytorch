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

        stats["epoch"] = self.epoch
        stats["iteration"] = iteration
        stats["mode"] = self.mode

        row = pd.DataFrame(stats, index=[self.idx])

        self.data = self.data.append(row, sort=False)
        self.idx +=1

    def get_data(self):
        return self.data

class VisdomLogger():
    def __init__(self):
        self.viz = Visdom()
        self.windows = dict()

        r = np.random.RandomState(0)
        self.colors = r.randint(0,255, size=(255,3))
        pass

    def update(self, data):
        data_mean_per_epoch = data.groupby(["mode", "epoch"]).mean()
        cols = data_mean_per_epoch.columns
        modes = data_mean_per_epoch.index.levels[0]

        for name in cols:

             if name in self.windows.keys():
                 win = self.windows[name]
                 update = 'new'
             else:
                 win = None # first log -> new window
                 update = None

             opts = dict(
                 title=name,
                 showlegend=True,
                 xlabel='epochs',
                 ylabel=name)

             for mode in modes:

                 epochs = data_mean_per_epoch[name].loc[mode].index
                 values = data_mean_per_epoch[name].loc[mode]

                 win = self.viz.line(
                     X=epochs,
                     Y=values,
                     name=mode,
                     win=win,
                     opts=opts,
                     update=update
                 )
                 update='insert'

             self.windows[name] = win

    def plot_images(self, target, output):

        # log softmax -> softmax
        output = np.exp(output)

        prediction = np.argmax(output, axis=1)

        target = np.swapaxes(self.colors[target], -1, 1)
        prediction = np.swapaxes(self.colors[prediction], -1, 1)

        self.viz.images(target, win="target", opts=dict(title='Target'))
        self.viz.images(prediction, win="predictions", opts=dict(title='Predictions'))

        b, c, h, w = output.shape
        for cl in range(c):
            arr = np.expand_dims(output[:,cl],1)*255
            self.viz.images(arr, win="class"+str(cl), opts=dict(title="class"+str(cl)))

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

