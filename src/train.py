import torch.nn
from utils.dataset import ijgiDataset as Dataset
from models.sequenceencoder import LSTMSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import sys
from utils.snapshot import save, resume
from visdom import Visdom




def main(
    datadir=sys.argv[1],
    batchsize = 16,
    shuffle = True,
    workers = 0,
    epochs = 100,
    lr = 1e-3,
    nclasses = 18,
    snapshot = "tmp.pth"
    ):

    traindataset = Dataset(datadir, tileids="tileids/train_fold0.tileids")
    testdataset = Dataset(datadir, tileids="tileids/test_fold0.tileids")

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)
    testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=shuffle,num_workers=workers)

    logger = Logger(columns=["loss", "test"], modes=["train", "test"])

    vizlogger = VisdomLogger()

    network = LSTMSequentialEncoder(48,48,nclasses=nclasses)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()



    #if snapshot is not None:
    #    state = resume(snapshot,model=network, optimizer=optimizer)

    for epoch in range(epochs):

        logger.update_epoch(epoch)

        print()
        print("Epoch {}\n".format(epoch))
        print("train")
        train_epoch(traindataloader, network, optimizer, loss, loggers=(logger,vizlogger))
        print("test")
        test_epoch(testdataloader, network,loss, loggers=(logger, vizlogger))

        data = logger.get_data()
        vizlogger.update(data)
        #data.groupby(["mode", "epoch"]).mean()

        checkpoint_path = "model_{:2d}.pth".format(epoch)
        save(checkpoint_path, network, optimizer)

def train_epoch(dataloader, network, optimizer, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")

    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target = data

        output = network.forward(input)
        l = loss(output, target)
        #print(l)
        stats = {"loss":l.data.cpu().numpy()}

        l.backward()
        optimizer.step()

        printer.print(stats, iteration)
        logger.log(stats, iteration)

        #if iteration > 5:
        #    return

def test_epoch(dataloader, network, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("test")

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target = data

            output = network.forward(input)
            l = loss(output, target)
            #print(l)
            stats = {"loss":l.data.cpu().numpy()}

            printer.print(stats, iteration)
            logger.log(stats, iteration)

            #if iteration > 5:
            #    break


        vizlogger.plot_images(target.cpu().detach().numpy(), output.cpu().detach().numpy())



if __name__ == "__main__":
    main()