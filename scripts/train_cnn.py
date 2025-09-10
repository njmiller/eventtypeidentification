import pickle
import datetime
import argparse
import os
import sys

import torch

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split

# from sklearn.metrics import confusion_matrix
import torchmetrics
import torchmetrics.classification

# from fit_cnn_model_binary import VoxelDataset, TestNet1, ComPairNet
from models.models import gen_testnet1, gen_testnet2
from datasets import AMEGOXVoxelDataset

class AccuracyLogits(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.zeros(4, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(4, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        # logits, target = self._input_format(logits, target)
        if logits.shape != target.shape:
            raise ValueError("logits and target must have same shape")

        logit_num = 1
        idx = logits <= -logit_num
        self.correct[0] += torch.sum(target[idx] == 0)
        self.total[0] += target[idx].numel()
        idx = torch.logical_and(logits > -logit_num, logits <= 0)
        self.correct[1] += torch.sum(target[idx] == 0) 
        self.total[1] += target[idx].numel()
        idx = torch.logical_and(logits > 0, logits <= logit_num)
        self.correct[2] += torch.sum(target[idx] == 1)
        self.total[2] += target[idx].numel()
        idx = logit_num < logits
        self.correct[3] += torch.sum(target[idx] == 1) 
        self.total[3] += target[idx].numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def blue(x): return '\033[94m' + x + '\033[0m'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(fn, extra=False):
    with open(fn, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    XBins, YBins, ZBins = 110, 110, 48
    XMin, XMax = -55, 55
    YMin, YMax = -55, 55
    ZMin, ZMax = 0, 48

    dims = [XBins, YBins, ZBins]
    xrange = [XMin, XMax]
    yrange = [YMin, YMax]
    zrange = [ZMin, ZMax]

    ranges = [xrange, yrange, zrange]

    # Split the dataset into training and validation datasets and make sure
    # the same seed is used for all processes.
    split = 0.9

    dataset_all = AMEGOXVoxelDataset(event_hits, event_types, dims, ranges, extra=extra)
    ntrain = int(len(dataset_all)*split)
    nval = len(dataset_all) - ntrain
    train_dataset, val_dataset = random_split(dataset_all, [ntrain, nval],
            generator=torch.Generator().manual_seed(42)
    )

    return train_dataset, val_dataset, dims

# Training code in the MNIST example
def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=50, dry_run=False):
    model.train()

    loss_train = 0
    sum_tot = 0

    num_world = torch.tensor(1).to(device)
    all_reduce(num_world)
    total_data = torch.tensor(0).to(device)
    
    time0 = datetime.datetime.now()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        sum_tot += torch.sum(data)
        data, target = data.to(device), target.to(device)

        num_data = torch.tensor(len(data)).to(device)
        all_reduce(num_data)
        total_data += num_data
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_fn(output, target.reshape((-1, 1)))
        loss_train += loss.item()*len(target)
        loss.backward()
        optimizer.step()
        time1 = datetime.datetime.now()
        time_elapsed = time1-time0
        
        if device == 0 and (batch_idx % log_interval == 0):
            percent_finished = 100. * batch_idx / len(train_loader)
            if percent_finished > 0:
                time_to_finish = time_elapsed * (100 - percent_finished) / percent_finished
            else:
                time_to_finish = time_elapsed * 10000
            print('                                                                                                   ', end='\r')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t{} : {}\tLoss: {:.6f}'.format(
                epoch, total_data.item(), len(train_loader.dataset),
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                percent_finished, time_elapsed, time_to_finish, loss.item()), end='\r')
            sys.stdout.flush()
            if dry_run:
                break
    
        
    loss_train /= len(train_loader.dataset)

    # print('')
    return loss_train


# Testing / Validation code
def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = torch.tensor(0).to(device)

    # cm = torch.zeros([2, 2]).to(device)
    # cm = torch.zeros([2, 2]).to(device)
    metric_acc = torchmetrics.classification.Accuracy(task="binary").to(device)
    metric_cm = torchmetrics.classification.ConfusionMatrix(task="binary").to(device)

    corr_bins = torch.zeros(11, 2).to(device)
    all_bins = torch.zeros(11, 2).to(device)
    bins = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    with torch.no_grad():
        for data, target, nhits in test_loader:
            target_cpu = target
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.view_as(output)).item()*len(target)

            pred = output.ge(0).type(torch.int)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # correct_ident = pred.eq(target.view_as(pred))
            # correct += correct_ident.sum().item()

            # Torchmetrics inputs are opposite order as scikit-learn
            # cm += torchmetrics.functional.confusion_matrix(pred, target.view_as(pred), task="binary")
            acc = metric_acc(pred, target.view_as(pred))
            cm = metric_cm(pred, target.view_as(pred))

            # Bins calculation. Need to figure out if there is a better way to do this than
            # iterating over all the data
            nhits_bins = torch.bucketize(nhits, bins)
            for bin_val, pred_tmp, target_tmp in zip(nhits_bins, pred, target.view_as(pred)):
                target_tmp = target_tmp.int()
                all_bins[bin_val, target_tmp] += 1
                if pred_tmp == target_tmp:
                    corr_bins[bin_val, target_tmp] += 1

    all_reduce(correct)
    correct = correct.cpu().item()

    acc = metric_acc.compute().item()
    cm = metric_cm.compute()

    rs_all = cm.diagonal()/cm.sum(axis=1)
    ps_all = cm.diagonal()/cm.sum(axis=0)

    rs_all = rs_all.tolist()
    ps_all = ps_all.tolist()

    all_reduce(all_bins)
    all_reduce(corr_bins)
    all_bins = all_bins.cpu()
    corr_bins = corr_bins.cpu()

    acc_com_bins = (corr_bins[:, 0] / all_bins[:, 0]).tolist()
    acc_pair_bins = (corr_bins[:, 1] / all_bins[:, 1]).tolist()

    test_loss /= len(test_loader.dataset)

    if device == 0:
        print('\nTest set: Loss: {:.4f}, Acc: {}/{} ({:.0f}%), Prec: {}, Rec: {}'.format(
              test_loss, correct, len(test_loader.dataset),
            #   100. * correct / len(test_loader.dataset),
              100. * acc,
              ps_all, rs_all))
        print("ACC COM:", acc_com_bins)
        print("ACC PAIR:", acc_pair_bins)
        print("N COM:", all_bins[:, 0].tolist())
        print("N PAIR:", all_bins[:, 1].tolist())

    return correct, test_loss

# Copied from MNIST example
def train_all(rank, model, params, train_dataset, test_dataset, dir='./', label=""):

    batch_size = params[0]
    n_epoch = params[1]
    rate_learning = params[2]
    outf = params[3]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, 
                              sampler=DistributedSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                             sampler=DistributedSampler(test_dataset))

    # Send the model to the correct GPU
    device = rank
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Is torch.compile worth it here?
    # model = torch.compile(model)

    # optimizer = optim.Adadelta(model.parameters(), lr=rate_learning)
    optimizer = optim.Adam(model.parameters(), lr=rate_learning)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Every 10 epochs multiply the learning rate by a factor of gamma
    # gamma = 0.5
    # scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # Loss function 
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    best_correct = 0
    loss_train = []
    loss_test = []
    correct_test = []
    for epoch in range(1, n_epoch + 1):
        train_loader.sampler.set_epoch(epoch)

        # Train and then append the avg of the average losses for each batch
        avg_loss_train = train(model, device, train_loader, optimizer, epoch, loss_fn)
        loss_train.append(avg_loss_train)

        # Run the test set and store the number of correct and the overall average loss
        curr_correct, avg_loss = test(model, device, test_loader, loss_fn)
        loss_test.append(avg_loss)
        correct_test.append(curr_correct)

        scheduler.step()

        # Save model if it is better than previous best model
        if rank == 0 and (curr_correct > best_correct):
            best_correct = curr_correct
            print("Saving new model, ncorrect =", best_correct)
            fn_state = "test_torch_model_params_" + label + ".pth"
            fn_state = os.path.join(dir, fn_state)
            torch.save(model.module.state_dict(), fn_state)
        print("")

    fn_loss = "loss_acc_values_" + label + ".txt"
    fn_loss = os.path.join(dir, fn_loss)
    with open(fn_loss, "w") as f:
        for list_tmp in (loss_train, loss_test, correct_test):
            for tmp in list_tmp:
                f.write(str(tmp))
                f.write(" ")
            f.write("\n")


def load_and_train(rank, world_size, fn, dir="./", label="", modelname='ComPairNet', batch_size=800,
                   nxy=110, rangexy=[-55, 55]):
    ddp_setup(rank, world_size)

    if rank == 0:
        print("Starting PyTorch training...")
        print("Loading data...", fn)

    with open(fn, 'rb') as f:
        event_hits, event_types = pickle.load(f)
    
    if rank == 0:
        print(f"Dataset contains {len(event_hits)} events")

        print(f"Batch size = {batch_size}")

    epochs = 20
    rate_learning = 0.001
    outf = 'torch_output'

    params = [batch_size, epochs, rate_learning, outf]

    if rank == 0:
        print("Initializing Torch Dataset...")
    
    # XBins, YBins, ZBins = 110, 110, 48
    # XBins, YBins, ZBins = 200, 200, 44
    # XMin, XMax = -45, 45
    # YMin, YMax = -45, 45
    
    XBins, YBins, ZBins = nxy, nxy, 44
    XMin, XMax = rangexy[0], rangexy[1]
    YMin, YMax = rangexy[0], rangexy[1]
    ZMin, ZMax = 0, 48 # Not used for AMEGO-X since z position has 44 unique values

    dims = [XBins, YBins, ZBins]
    xrange = [XMin, XMax]
    yrange = [YMin, YMax]
    zrange = [ZMin, ZMax]

    ranges = [xrange, yrange, zrange]

    print("Ranges:", ranges)
    print("NBins:", dims)

    # Split the dataset into training and validation datasets and make sure
    # the same seed is used for all processes.
    split = 0.9

    # dataset_all = VoxelDataset(event_hits, event_types, dims, ranges)
    dataset_all = AMEGOXVoxelDataset(event_hits, event_types, dims, ranges, extra=True, experiment="AMEGOX")

    # nsub = 3000
    # dataset_all, _ = random_split(dataset_all, [nsub, len(dataset_all)-nsub])
    
    ntrain = int(len(dataset_all)*split)
    nval = len(dataset_all) - ntrain

    train_dataset, val_dataset = random_split(dataset_all, [ntrain, nval],
            generator=torch.Generator().manual_seed(42)
    )

    # Make an instance of the correct model defined in fit_cnn_model_binary.py
    if modelname == 'ComPairNet':
        print("Initializing PyTorch binary classification version of ComPairNet...")
        # model = ComPairNet(input_shape=(XBins, YBins, ZBins))
        raise ValueError("Don't have ComPairNet right now")
    elif modelname == 'TestNet1':
        if rank == 0:
            print("Initializing PyTorch binary classification version of TestNet1...")
        # model = TestNet1(input_shape=(XBins, YBins, ZBins))
        model = gen_testnet1(input_shape=(XBins, YBins, ZBins))
    elif modelname == "TestNet2":
        if rank == 0:
            print("Initializing PyTorch binary classification version of model suggested by AI...")
        model = gen_testnet2(input_shape=(XBins, YBins, ZBins))
    else:
        raise ValueError("Bad modelname")

    time0 = datetime.datetime.now()
    train_all(rank, model, params, train_dataset, val_dataset, dir=dir, label=label)
    time1 = datetime.datetime.now()
    if rank == 0:
        print("Time to run:", time1-time0)

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Classification of Events')

    parser.add_argument('-fn', dest='fn', action='store', help='Dataset filename')
    parser.add_argument('-label', dest='label', action='store', default="",
                        help='Label to add to output data')
    parser.add_argument('-dir', dest='dir', action='store', default="./",
                        help='Directory for output data')
    parser.add_argument('-model', dest='model', action='store', default='ComPairNet',
                        help='Model to use')
    parser.add_argument("-batch", dest='batch', action='store', type=int, default=800,
                        help="Batch size")
    parser.add_argument("-nxy", dest='nxy', action='store', type=int, default=110,
                        help="Number of bins in X/Y direction")
    parser.add_argument("-rangexy", dest='rangexy', action='store', nargs='+', type=int, default=[-55, 55],
                        help="Range for bins (Single value instead of list has the code figure out the limits)")
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    args_input=(world_size, args.fn, args.dir, args.label, args.model, args.batch, args.nxy, args.rangexy)
    # mp.spawn(load_and_train, args=(world_size, args.fn, args.dir, args.label, args.model, args.batch,
                                    # args.nxy, args.rangexy),
    mp.spawn(load_and_train, args=args_input, nprocs=world_size)
    # load_and_train(args.fn, dir=args.dir, label=args.label, modelname=args.model, batch_size=args.batch)