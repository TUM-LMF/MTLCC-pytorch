import torch.nn
from utils.dataset import ijgiDataset as Dataset
from models.sequenceencoder import LSTMSequentialEncoder
import visdom
import sys

traindir=sys.argv[1]
testdir=None
batchsize=3
shuffle=True
workers=1
epochs=3
lr=1e-3

traindataset = Dataset(traindir)
#testdataset = Dataset(testdir)

traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)
#testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)

network = LSTMSequentialEncoder(48,48)

network = torch.nn.DataParallel(network).cuda()

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
loss=torch.nn.NLLLoss().cuda()

for epoch in range(epochs):

    for iteration, data in enumerate(traindataloader):
        optimizer.zero_grad()

        input, target = data

        output = network.forward(input)
        l = loss(output, target.cuda())
        #print(l)
        print("step")

        l.backward()
        optimizer.step()

