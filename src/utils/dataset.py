import torch.utils.data
import os
import rasterio
import torch
import numpy as np
import torch.nn.functional as F
import random
from utils.progressbar import ProgressBar

LABEL_FILENAME="y.tif"

def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples=range(100)
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #return self.samples[idx]
        tile = "generated"
        input = torch.randn((20, 13, 24, 24))
        target = torch.randint(0, 7, (24,24), dtype=torch.long)

        return input, target


class ijgiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, seqlength=30, tileids=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir,"data")
        self.seqlength=seqlength

        stats=dict(
            rejected_nopath=0,
            rejected_length=0,
            total_samples=0)

        # statistics
        self.samples = list()

        self.ndates = list()


        if tileids is None:
            files = os.listdir(self.data_dir)
        else:
            # tileids e.g. "tileids/train_fold0.tileids" path of line separated tileids specifying
            with open(os.path.join(root_dir, tileids), 'r') as f:
                files = [el.replace("\n", "") for el in f.readlines()]

        self.classids, self.classes = self.read_classes(os.path.join(self.root_dir, "classes.txt"))

        progress = ProgressBar(len(files), fmt=ProgressBar.FULL)

        for f in files:
            progress.current += 1
            progress()


            path = os.path.join(self.data_dir, f)


            if not os.path.exists(path):
                stats["rejected_nopath"] += 1
                continue

            #read(join(path,"y.tif"))
            ndates = len(get_dates(path))
            if ndates < self.seqlength:
                stats["rejected_length"] += 1
                continue # skip shorter sequence lengths

            # save class histogram
            #arr, _ = read(os.path.join(path, LABEL_FILENAME))
            #self.labelhistograms.append(np.histogram(arr, bins=range(arr.max())))

            stats["total_samples"] += 1
            self.samples.append(f)
            self.ndates.append(ndates)


        progress.done()

        print_stats(stats)

    def read_classes(self, csv):

        with open(csv, 'r') as f:
            classes = f.readlines()

        ids=list()
        names=list()
        for row in classes:
            row=row.replace("\n","")
            if '|' in row:
                id,cl = row.split('|')
                ids.append(int(id))
                names.append(cl)

        return ids, names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path = os.path.join(self.data_dir, self.samples[idx])

        label, profile = read(os.path.join(path,LABEL_FILENAME))

        profile["name"] = self.samples[idx]

        # unique dates sorted ascending
        dates = get_dates(path, n=self.seqlength)

        x10 = list()
        x20 = list()
        x60 = list()

        for date in dates:

            x10.append(read(os.path.join(path, date + "_10m.tif"))[0])
            x20.append(read(os.path.join(path, date + "_20m.tif"))[0])
            x60.append(read(os.path.join(path, date + "_60m.tif"))[0])

        x10 = np.array(x10) * 1e-4
        x20 = np.array(x20) * 1e-4
        x60 = np.array(x60) * 1e-4

        # augmentation
        # if np.random.rand() < self.augmentrate:
        #     x10 = np.fliplr(x10)
        #     x20 = np.fliplr(x20)
        #     x60 = np.fliplr(x60)
        #     label = np.fliplr(label)
        # if np.random.rand() < self.augmentrate:
        #     x10 = np.flipud(x10)
        #     x20 = np.flipud(x20)
        #     x60 = np.flipud(x60)
        #     label = np.flipud(label)
        # if np.random.rand() < self.augmentrate:
        #     angle = np.random.choice([1, 2, 3])
        #     x10 = np.rot90(x10, angle, axes=(2, 3))
        #     x20 = np.rot90(x20, angle, axes=(2, 3))
        #     x60 = np.rot90(x60, angle, axes=(2, 3))
        #     label = np.rot90(label, angle, axes=(0, 1))

        # replace stored ids with index in classes csv
        #label =
        label=label[0]
        new = np.zeros(label.shape, np.int)
        for cl,i in zip(self.classids, range(len(self.classids))):
            new[label == cl] = i

        label=new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        x20 = torch.from_numpy(x20)
        x60 = torch.from_numpy(x60)

        x20 = F.interpolate(x20, size=x10.shape[2:4])
        x60 = F.interpolate(x60, size=x10.shape[2:4])

        x = torch.cat((x10, x20, x60), 1)




        #x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, npad), mode='constant', value=-1)

        return x.float(), label.long()

def get_dates(path, n=None):
    """
    extracts a list of unique dates from dataset sample

    :param path: to dataset sample folder
    :param n: choose n random samples from all available dates
    :return: list of unique dates in YYYYMMDD format
    """

    files = os.listdir(path)
    dates = list()
    for f in files:
        f = f.split("_")[0]
        if len(f) == 8:  # 20160101
            dates.append(f)

    dates = set(dates)

    if n is not None:
        dates = random.sample(dates, n)

    dates = list(dates)
    dates.sort()
    return dates

def print_stats(stats):
    print_lst = list()
    for k,v in zip(stats.keys(), stats.values()):
        print_lst.append("{}:{}".format(k,v))

    print(", ".join(print_lst))

if __name__=="__main__":

    dataset = ijgiDataset("/data/datasets/ijgi2018_tif/480/data16")
    a = dataset[0]
