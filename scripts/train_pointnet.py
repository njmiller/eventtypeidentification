import pickle
import datetime
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce, all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split

# from sklearn.metrics import confusion_matrix
import torchmetrics
import torchmetrics.classification

# from fit_cnn_model_binary import VoxelDataset, TestNet1, ComPairNet
from models.pointnet import PointNet, PointNetLoss
from data.datasets import AMEGOXPointCloud, pc_collate_fn


MASTER_PORT = "12355"
TRAIN_VAL_SPLIT = 0.9
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.001
LOG_INTERVAL = 50

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gather_and_concat(data, world_size=2):

    device = data.device 
    data_all = [torch.zeros_like(data, device=device) for _ in range(world_size)]

    all_gather(data_all, data)

    data_all = torch.cat(data_all, dim=0)

    return data_all

def load_dataset(fn, extra=False):
    with open(fn, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    # Split the dataset into training and validation datasets and make sure
    # the same seed is used for all processes.
    dataset_all = AMEGOXPointCloud(event_hits, event_types)

    ntrain = int(len(dataset_all)*TRAIN_VAL_SPLIT)
    nval = len(dataset_all) - ntrain
    train_dataset, val_dataset = random_split(dataset_all, [ntrain, nval],
            generator=torch.Generator().manual_seed(DEFAULT_SEED)
    )

    return train_dataset, val_dataset

# Training code in the MNIST example
def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=50, dry_run=False):
    model.train()

    loss_train = 0

    num_world = torch.tensor(1).to(device)
    all_reduce(num_world)
    total_data = torch.tensor(0).to(device)
    
    time0 = datetime.datetime.now()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        num_data = torch.tensor(len(data)).to(device)
        all_reduce(num_data)
        total_data += num_data
        
        optimizer.zero_grad()
        output, trans_feat = model(data)

        # loss = loss_fn(output, target.reshape((-1, 1)))
        loss = loss_fn(output, target.reshape((-1, 1)), trans_feat)

        loss_train += loss.item()*len(target)
        loss.backward()
        optimizer.step()
        time1 = datetime.datetime.now()
        time_elapsed = time1-time0
        
        if device == 0 and (batch_idx % log_interval == 0):
            percent = 100. * batch_idx / len(train_loader)
            eta = time_elapsed * (100-percent) / percent if percent > 0 else datetime.timedelta(hours=10)
            print(f'\rTrain Epoch: {epoch} [{total_data.item()}/{len(train_loader.dataset)} '
                  f'({percent:.0f}%)] Loss: {loss.item():.6f} ETA: {eta}', end='', flush=True)
            if dry_run:
                break
    
        
    loss_train /= len(train_loader.dataset)

    return loss_train


# Testing / Validation code
def test(model, device, test_loader, loss_fn, epoch):
    model.eval()
    test_loss = 0
    correct = torch.tensor(0).to(device)

    metric_acc = torchmetrics.classification.Accuracy(task="binary").to(device)
    metric_cm = torchmetrics.classification.ConfusionMatrix(task="binary").to(device)

    probs = torch.tensor([], device=device)
    targets = torch.tensor([], device=device)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, trans_feat = model(data)

            test_loss += loss_fn(output, target.view_as(output), trans_feat).item()*len(target)

            pred = output.ge(0).type(torch.int)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Torchmetrics inputs are opposite order as scikit-learn
            # cm += torchmetrics.functional.confusion_matrix(pred, target.view_as(pred), task="binary")
            acc = metric_acc(pred, target.view_as(pred))
            cm = metric_cm(pred, target.view_as(pred))

            probs = torch.cat((probs, F.sigmoid(output)))
            targets = torch.cat((targets, target))

    probs = probs.view_as(targets)    

    probs_all = gather_and_concat(probs).cpu()
    targets_all = gather_and_concat(targets).cpu()

    all_reduce(correct)
    correct = correct.cpu().item()

    acc = metric_acc.compute().item()
    cm = metric_cm.compute()

    rs_all = cm.diagonal()/cm.sum(axis=1)
    ps_all = cm.diagonal()/cm.sum(axis=0)

    rs_all = rs_all.tolist()
    ps_all = ps_all.tolist()

    test_loss /= len(test_loader.dataset)

    if device == 0:
        percent = 100. * acc
        print(f'\nTest set: Loss: {test_loss:.4f}, Acc: {correct}/{len(test_loader.dataset)} '
              f'({percent:.0f}%), Prec: {ps_all}, Rec: {rs_all}')
        # fn = "probs_targets_epoch"+str(epoch)+".pt"
        # torch.save({ 
            # 'probs': probs_all,
            # 'targets': targets_all
        # }, fn)

    return correct, test_loss

# Copied from MNIST example
def train_all(rank, model, params, train_dataset, test_dataset, dir='./', label=""):

    batch_size = params[0]
    n_epoch = params[1]
    rate_learning = params[2]
    outf = params[3]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, 
                              sampler=DistributedSampler(train_dataset), collate_fn=pc_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                             sampler=DistributedSampler(test_dataset), collate_fn=pc_collate_fn)

    # Send the model to the correct GPU
    device = rank
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # optimizer = optim.Adadelta(model.parameters(), lr=rate_learning)
    optimizer = optim.Adam(model.parameters(), lr=rate_learning)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Every 10 epochs multiply the learning rate by a factor of gamma
    # gamma = 0.5
    # scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # Loss function 
    # loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    loss_fn = PointNetLoss(F.binary_cross_entropy_with_logits).to(device)

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
        curr_correct, avg_loss = test(model, device, test_loader, loss_fn, epoch)
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

            # Put in a specific tensor to make sure model is loaded correctly in different code
            test1 = torch.zeros([1, 4, 3])
            test1[0, :, 0] = 1
            test1[0, :, 1] = 2
            test1[0, :, 2] = 2.5
            model.eval()
            with torch.no_grad():
                logits, trans_feat = model(test1)
                print("VALS FOR C++ TEST:", logits.item(), torch.sum(trans_feat).item())
            model.train()
        print("")

    fn_loss = "loss_acc_values_" + label + ".txt"
    fn_loss = os.path.join(dir, fn_loss)
    with open(fn_loss, "w") as f:
        for list_tmp in (loss_train, loss_test, correct_test):
            for tmp in list_tmp:
                f.write(str(tmp))
                f.write(" ")
            f.write("\n")


def load_and_train(rank, world_size, fn, dir="./", label="", batch_size=800, weights=None):
    ddp_setup(rank, world_size)

    if rank == 0:
        print("Starting PyTorch training...")
        print("Loading data...", fn)

    if rank == 0:
        print(f"Batch size = {batch_size}")

    epochs = DEFAULT_EPOCHS
    rate_learning = DEFAULT_LR
    outf = 'torch_output'

    params = [batch_size, epochs, rate_learning, outf]

    if rank == 0:
        print("Initializing Torch Dataset...")
    
    # Load the training and validation datasets
    train_dataset, val_dataset = load_dataset(fn)

    if rank == 0:
        print(f"Training dataset contains {len(train_dataset)} events")
        print(f"Validation dataset contains {len(val_dataset)} events")


    model = PointNet(add_nhits=False)
    if weights is not None:
        model.load_state_dict(torch.load(weights, weights_only=True))

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
    parser.add_argument("-batch", dest='batch', action='store', type=int, default=800,
                        help="Batch size")
    parser.add_argument("-weights", dest='weights', action='store',
                        help="Initial weights")
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    args_input=(world_size, args.fn, args.dir, args.label, args.batch, args.weights)
    mp.spawn(load_and_train, args=args_input, nprocs=world_size)