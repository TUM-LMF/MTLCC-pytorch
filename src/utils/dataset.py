import torch.utils.data

class Dataset(torch.utils.data.Dataset):
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

        return tile, input, target
