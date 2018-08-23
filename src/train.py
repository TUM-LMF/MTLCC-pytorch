import torch.nn
from utils.dataset import ijgiDataset as Dataset
from models.sequenceencoder import LSTMSequentialEncoder
import visdom
import sys

traindir=sys.argv[1]
testdir=None
batchsize=16
shuffle=True
workers=16
epochs=100
lr=1e-3
nclasses=18


traindataset = Dataset(traindir)
#testdataset = Dataset(testdir)

traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)
#testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)

network = LSTMSequentialEncoder(48,48,nclasses=nclasses)

network = torch.nn.DataParallel(network).cuda()

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
loss=torch.nn.NLLLoss().cuda()

for epoch in range(epochs):

    for iteration, data in enumerate(traindataloader):
        optimizer.zero_grad()

        input, target = data

        output = network.forward(input)
        l = loss(output, target.cuda()).cuda()
        #print(l)
        print(l.data.cpu().numpy())

        l.backward()
        optimizer.step()

