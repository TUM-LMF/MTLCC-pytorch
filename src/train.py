import torch.nn
from utils.dataset import Dataset
from models.sequenceencoder import LSTMSequentialEncoder
import visdom

traindir=None
testdir=None
batchsize=1
shuffle=True
workers=8
epochs=3
lr=1e-3

traindataset = Dataset(traindir)
testdataset = Dataset(testdir)

traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)
testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)

network = LSTMSequentialEncoder(24,24)
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
loss=torch.nn.NLLLoss()

for epoch in range(epochs):

    for iteration, data in enumerate(traindataloader):
        optimizer.zero_grad()

        id, input, target = data

        output = network.forward(input)
        l = loss(output, target)

        l.backward()
        optimizer.step()

