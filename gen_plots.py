import pickle

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import torch_models as tm

def make_plot(data, target, pred, i):
    data = np.array(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    amp = data[:, 3] / np.max(data[:, 3])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=amp, cmap='coolwarm')
    ax.set_title(f'Target: {target}, Pred: {pred}')

    if target == pred:
        cf = "correct"
    else:
        cf = "wrong"
    fn = f'images/{i}_target_{target}_{cf}.png'
    fig.savefig(fn)
    plt.close(fig)

def infer_and_plot():

    fn = 'test_torch_model.pth'
    model = tm.VoxNet(num_classes=2, input_shape=(110, 110, 48))
    model.load_state_dict(torch.load(fn))
    model.eval()

    fn_data = '/data/slag2/njmille2/test_dataset_nhits12.pkl'

    with open(fn_data, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    print("SSS:", len(event_types))
    print("TTT:", np.sum(np.array(event_types) == 0))
    print("UUU:", np.sum(np.array(event_types) == 1))
    
    event_hits = event_hits[:400]
    event_types = event_types[:400]

    XBins, YBins, ZBins = 110, 110, 48
    XMin, XMax = -55, 55
    YMin, YMax = -55, 55
    ZMin, ZMax = 0, 48

    dims = [XBins, YBins, ZBins]
    xrange = [XMin, XMax]
    yrange = [YMin, YMax]
    zrange = [ZMin, ZMax]

    ranges = [xrange, yrange, zrange]
    
    dataset = tm.VoxelDataset(event_hits, event_types, dims, ranges)
    
    batch_size = 400
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda")
    device_cpu = torch.device("cpu")

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    pred_all = []
    target_all = []
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_all = np.append(pred_all, pred.cpu())
            target_all = np.append(target_all, target.cpu())
            
            # output_cpu = output.to(device_cpu)
            # print("AAA:", output_types)
            # print("BBB:", output_cpu)
            # output_types = output_types + output

    for i, tmp in enumerate(dataset):
        voxel, target = tmp
        data = dataset.data[i] #get the actual hit data
        pred = int(pred_all[i])
        target = int(target)
        print("SSS:", i, np.shape(voxel), target, pred, np.shape(data))
        make_plot(data, target, pred, i)

    # print("Pred:", pred_all)
    # print("Target:", target_all)
    # cm = tm.model_to_cm(model, device, dataset_loader)
    # print("CM:", cm)
    
    # rs = cm.diagonal()/cm.sum(axis=1)
    # ps = cm.diagonal()/cm.sum(axis=0)

    # print("Recall:", rs)
    # print("Precision:", ps)


if __name__ == '__main__':
    infer_and_plot()