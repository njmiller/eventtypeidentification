import argparse
import os

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce, all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split, DataLoader

from torchinfo import summary

import torchmetrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fit_cnn_model_binary_distributed import load_dataset, AccuracyLogits
from fit_cnn_model_binary import TestNet1, VoxelDataset

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

# Testing / Validation code
def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = torch.tensor(0).to(device)

    metric_acc = torchmetrics.classification.Accuracy(task="binary").to(device)
    metric_cm = torchmetrics.classification.ConfusionMatrix(task="binary").to(device)
    metric_acc2 = AccuracyLogits().to(device)

    logits_all = []
    nhits_all = []
    target_all = []
    energy_all = []
    bincount_all = []

    if device == 0:
        print("Starting inference on validation set")

    with torch.no_grad():
        for data, target, nhits in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.view_as(output)).item()*len(target)

            pred = output.ge(0).type(torch.int)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Torchmetrics code
            acc = metric_acc(pred, target.view_as(pred))
            cm = metric_cm(pred, target.view_as(pred))
            acc2 = metric_acc2(output, target.view_as(output))

            # Extra testing stuff
            nhits_all.append(nhits.to(device))
            logits_all.append(output)
            target_all.append(target)
            energy_all.append(data.sum(dim=(1, 2, 3, 4)))

            tmp = data != 0
            bincount_all.append(tmp.sum(dim=(1, 2, 3, 4)))

    # nhits_all = gather_all(nhits_all)
    # logits_all = gather_all(logits_all)

    nhits_all = torch.cat(nhits_all)
    logits_all = torch.cat(logits_all)
    target_all = torch.cat(target_all)
    energy_all = torch.cat(energy_all)
    bincount_all = torch.cat(bincount_all)

    '''
    nhits_all_list = [torch.zeros(128, dtype=torch.int64, device=device) for _ in range(2)]
    logits_all_list = [torch.zeros(128, dtype=torch.float32, device=device) for _ in range(2)]
    all_gather(nhits_all_list, nhits_all)
    all_gather(logits_all_list, logits_all)
    
    nhits_all = torch.cat(nhits_all_list)
    logits_all = torch.cat(logits_all_list)
    '''

    all_reduce(correct)
    correct = correct.cpu().item()

    acc = metric_acc.compute().item()
    cm = metric_cm.compute()
    acc2 = metric_acc2.compute().tolist()

    rs_all = cm.diagonal()/cm.sum(axis=1)
    ps_all = cm.diagonal()/cm.sum(axis=0)

    rs_all = rs_all.tolist()
    ps_all = ps_all.tolist()

    test_loss /= len(test_loader.dataset)

    if device == 0:
        print('\nTest set: Loss: {:.4f}, Acc: {}/{} ({:.0f}%), Prec: {}, Rec: {}\n'.format(
              test_loss, correct, len(test_loader.dataset),
            #   100. * correct / len(test_loader.dataset),
              100. * acc,
              ps_all, rs_all))
        print('Acc2:', acc2)
        # print("TEST:", acc)

    if device == 0:
        logits_all, nhits_all = logits_all.cpu(), nhits_all.cpu()
        target_all = target_all.cpu()
        energy_all = energy_all.cpu()
        bincount_all = bincount_all.cpu()

        logits_all = torch.reshape(logits_all, shape=(-1, ))
        nhits_all = torch.reshape(nhits_all, shape=(-1, ))
        target_all = torch.reshape(target_all, shape=(-1, ))
        
        plt.figure()
        plt.scatter(logits_all, nhits_all)
        plt.savefig("logits_vs_nhits.png")
        plt.close()

        plt.figure()
        plt.scatter(target_all, nhits_all)
        plt.savefig("target_vs_nhits.png")
        plt.close()

        idx0 = target_all == 0
        idx1 = target_all == 1

        plt.figure()
        plt.hist(logits_all[idx0], bins=torch.linspace(-10, 3, 20))
        plt.savefig("hist_logits_target0.png")
        plt.close()
        
        plt.figure()
        plt.hist(logits_all[idx1], bins=torch.linspace(-3, 10, 20))
        plt.savefig("hist_logits_target1.png")
        plt.close()

        plt.figure()
        plt.scatter(logits_all[idx0], nhits_all[idx0], c='b')
        plt.scatter(logits_all[idx1], nhits_all[idx1], c='r')
        plt.savefig("logits_vs_nhits2.png")
        plt.close()
        
        plt.figure()
        plt.scatter(logits_all[idx1], nhits_all[idx1], c='r')
        plt.scatter(logits_all[idx0], nhits_all[idx0], c='b')
        plt.savefig("logits_vs_nhits3.png")
        plt.close()
        
        plt.figure()
        plt.scatter(logits_all[idx0], energy_all[idx0], c='b')
        plt.scatter(logits_all[idx1], energy_all[idx1], c='r')
        plt.savefig("logits_vs_energy.png")
        plt.close()
        
        plt.figure()
        plt.scatter(logits_all[idx1], energy_all[idx1], c='r')
        plt.scatter(logits_all[idx0], energy_all[idx0], c='b')
        plt.savefig("logits_vs_energy2.png")
        plt.close()
        
        plt.figure()
        plt.scatter(logits_all[idx0], bincount_all[idx0], c='b')
        plt.scatter(logits_all[idx1], bincount_all[idx1], c='r')
        plt.savefig("logits_vs_bincount.png")
        plt.close()
        
        plt.figure()
        plt.scatter(logits_all[idx1], bincount_all[idx1], c='r')
        plt.scatter(logits_all[idx0], bincount_all[idx0], c='b')
        plt.savefig("logits_vs_bincount2.png")
        plt.close()
        
        plt.figure()
        plt.hist(bincount_all[idx0], bins=torch.arange(1, 102, 2))
        plt.savefig("Bins_compton2.png")
        plt.close()
        
        plt.figure()
        plt.hist(bincount_all[idx1], bins=torch.arange(1, 102, 2))
        plt.savefig("Bins_pair2.png")
        plt.close()
 
    return correct, test_loss

def load_and_test(rank, world_size, fn, dir="./", label="", batch_size=800):
    ddp_setup(rank, world_size)

    device = rank

    if rank == 0:
        print("Starting PyTorch model testing...")
        print("Loading data...", fn)

    _, test_dataset, dims = load_dataset(fn, extra=True)

    if rank == 0:
        print(f"Test dataset contains {len(test_dataset)} events")
        print(f"Batch size = {batch_size}")


    model = TestNet1(input_shape=tuple(dims))
    fn = os.path.join(dir, "test_torch_model_params_" + label + ".pth")
    model.load_state_dict(torch.load(fn))
    model.eval()

    if rank == 0:
        summary(model)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                             sampler=DistributedSampler(test_dataset))
    
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    
    test(model, device, test_loader, loss_fn)

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Classification of Events')

    parser.add_argument('-fn', dest='fn', action='store', help='Dataset filename')
    parser.add_argument('-label', dest='label', action='store', default="",
                        help='Label for saved model')
    parser.add_argument('-dir', dest='dir', action='store', default="./",
                        help='Directory to find data')
    parser.add_argument("-batch", dest='batch', action='store', type=int, default=800,
                        help="Batch size")
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(load_and_test, args=(world_size, args.fn, args.dir, args.label, args.batch),
             nprocs=world_size)