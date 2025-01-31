import pickle

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import VoxelDataset, gen_testnet1

def make_plot(data, target, pred, i, nhits, prob):
    data = np.array(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    amp = data[:, 3] / np.max(data[:, 3])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=amp, cmap='coolwarm')
    ax.set_title(f'Target: {target}, Pred: {pred}, NHits: {nhits}, Prob : {prob:.2f}', {"fontsize": 12})

    if target == pred:
        cf = "correct"
    else:
        cf = "wrong"

    fn = f'images/{i}_target_{target}_{cf}_{nhits}_nhits.png'

    if 30 < nhits < 70 and target != pred:
        fig.savefig(fn)

    plt.close(fig)

def infer_and_plot():

    # fn = 'test_torch_model.pth'
    fn = '/data/slag2/njmille2/test_torch_model_params_TestFunc.pth'
    # model = tm.VoxNet(num_classes=2, input_shape=(110, 110, 48))
    model = gen_testnet1()
    model.load_state_dict(torch.load(fn))
    model.eval()

    # fn_data = '/data/slag2/njmille2/test_dataset_nhits12.pkl'
    fn_data = '/data/slag2/njmille2/test_dataset_nhits2_detector1_2500000.pkl'
    with open(fn_data, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    nevents = len(event_types)
    nevents0 = np.sum(np.array(event_types) == 0)
    nevents1 = np.sum(np.array(event_types) == 1)

    print("Num Events:", nevents)
    print("Num Type 0:", nevents0)
    print("Num Type 1:", nevents1)

    # Get a small subset of events of each type
    # Input data is not sorted.
    nsub = 1000 
    # nsub = 10
    event_hits0 = event_hits[:nsub]
    event_types0 = event_types[:nsub]

    event_hits1 = event_hits[nevents0:nevents0+nsub]
    event_types1 = event_types[nevents0:nevents0+nsub]

    event_hits = event_hits0 + event_hits1
    event_types = event_types0 + event_types1

    XBins, YBins, ZBins = 110, 110, 48
    XMin, XMax = -55, 55
    YMin, YMax = -55, 55
    ZMin, ZMax = 0, 48

    dims = [XBins, YBins, ZBins]
    xrange = [XMin, XMax]
    yrange = [YMin, YMax]
    zrange = [ZMin, ZMax]

    ranges = [xrange, yrange, zrange]
    
    dataset = VoxelDataset(event_hits, event_types, dims, ranges, extra=True)
    
    batch_size = 128
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda")
    # device_cpu = torch.device("cpu")

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Run through inference and get prediction for the subset of data
    pred_all = []
    prob_all = []
    with torch.no_grad():
        for data, target, _ in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = output.ge(0).type(torch.int)
            pred_all = np.append(pred_all, pred.cpu())

            prob = torch.nn.functional.sigmoid(output)
            prob_all = np.append(prob_all, prob.cpu())

    for i, tmp in enumerate(dataset):
        _, target, nhits = tmp
        data = dataset.data[i] #get the actual hit data
        pred = int(pred_all[i])
        prob = prob_all[i]
        target = int(target)
        # print("SSS:", i, np.shape(voxel), target, pred, np.shape(data))
        make_plot(data, target, pred, i, nhits, prob)

if __name__ == '__main__':
    infer_and_plot()