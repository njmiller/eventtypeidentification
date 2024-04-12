import pickle

import numpy as np

import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import confusion_matrix, precision_score, recall_score

def blue(x): return '\033[94m' + x + '\033[0m'

class VoxelDataset(Dataset):
    def __init__(self, data, labels, dims, ranges):
        self.data = data  # x, y, z, e for each hit in each event
        self.labels = labels  # type label number for each event

        labels_unique = np.unique(labels)

        self.nlabels = len(labels_unique)
        self.label_map = {}
        for i, val in enumerate(labels_unique):
            self.label_map[val] = i

        # Info related to converting the point cloud to a voxel tensor
        self.dims = dims
        self.xbins = dims[0]
        self.ybins = dims[1]
        self.zbins = dims[2]

        xrange = ranges[0]
        yrange = ranges[1]
        zrange = ranges[2]

        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.ymin = yrange[0]
        self.ymax = yrange[1]
        self.zmin = zrange[0]
        self.zmax = zrange[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Generate the tensor for this idx"""

        # Get the hits and the label for this event
        data_idx = self.data[idx]
        label_idx = self.labels[idx]

        tensor = torch.zeros([1, self.xbins, self.ybins, self.zbins])
        # nhits = len(data_idx)

        # Iterate over all the hits, find the bin for each hit and add its energy to
        # that bin
        for i in data_idx:
            xbin = (int)(((i[0] - self.xmin) / (self.xmax - self.xmin)) * self.xbins)
            ybin = (int)(((i[1] - self.ymin) / (self.ymax - self.ymin)) * self.ybins)
            zbin = (int)(((i[2] - self.zmin) / (self.zmax - self.zmin)) * self.zbins)
            tensor[0, xbin, ybin, zbin] += i[3]

        # We return the tensor and the label
        # if label_idx == 12:
            # label_out = 1
        # else:
            # label_out = 0

        # Separating into Pair (10-19) and Compton (0-9)
        if label_idx >= 10:
            label_out = 1
        else:
            label_out = 0

        # return tensor, self.label_map[label_idx]
        return tensor, label_out

class VoxNet2(torch.nn.Module):

    def __init__(self):
        pass


class VoxNet(torch.nn.Module):

    def __init__(self, num_classes=10, input_shape=(32, 32, 32)):
                 #weights_path=None,
                 #load_body_weights=True,
                 #load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.

        Modified in order to accept different input shapes.

        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True

        Notes
        -----
        Weights available at: url to be added

        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.body = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv3d(in_channels=1,
                                      out_channels=32, kernel_size=5, stride=2)),
            ('lkrelu1', torch.nn.LeakyReLU(0.1)),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('lkrelu2', torch.nn.LeakyReLU(0.1)),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        # Trick to accept different input shapes. Just puts a random
        # variable through the body and checks its output dimensions
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(OrderedDict([
            ('flat', torch.nn.Flatten()),
            ('fc1', torch.nn.Linear(first_fc_in_features, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, num_classes)),
            # ('out1', torch.nn.Softmax(dim=1))
            ('out1', torch.nn.LogSoftmax(dim=1))
        ]))

        #if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, x):
        x = self.body(x)
        # x = x.view(x.size(0), -1)
        # x = x.reshape(x.size(0), -1)
        # x = torch.flatten(x)
        x = self.head(x)
        return x

def train2(model, params, train_dataset, test_dataset):

    batch_size = params[0]
    n_epoch = params[1]
    rate_learning = params[2]
    outf = params[3]

    use_cuda = True
    use_mps = False
    if use_cuda & torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps & torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    num_batch = len(train_dataset) / batch_size
    # print(num_batch)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in range(n_epoch):
        # scheduler.step()
        for i, (voxel, cls_idx) in enumerate(train_dataloader, 0):
            voxel, cls_idx = voxel.to(device), cls_idx.to(device)
            # voxel = voxel.float()

            optimizer.zero_grad()

            model = model.train()
            pred = model(voxel)

            # loss = F.cross_entropy(pred, cls_idx)
            loss = F.nll_loss(pred, cls_idx)

            loss.backward()
            optimizer.step()

            # max(1) is maximum along dim=1
            # the returned value is a tuple of (value, idxs) of which we want the indices
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' %
                  (epoch, i, num_batch, loss.item(), correct.item() / float(batch_size)))

            # Every 5 batches run on the testing set and output the loss and accuracy
            # if i % 5 == 0:
                # j, sample = next(enumerate(test_dataloader, 0))

                # voxel, cls_idx = sample
                # voxel, cls_idx = voxel.to(device), cls_idx.to(device)

                # Get the class numbers for each input
                # cls_idx = cls_idx.data.max(1)[1]

                # voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])

                # model = model.eval()
                # pred = model(voxel)

                #loss = F.nll_loss(pred, cls_idx)
                # loss = F.cross_entropy(pred, cls_idx)

                # pred_choice = pred.data.max(1)[1]

                # correct = pred_choice.eq(cls_idx.data).cpu().sum()
                # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                                # blue('test'), loss.item(),
                                                                # correct.item() / float(batch_size)))


        test(model, device, test_dataloader)
        # Save model. Maybe figure out how to save only if the model has improved
        # torch.save(model.state_dict(), '%s/cls_model_%d.pth' % (outf, epoch))


# Training code in the MNIST example
def train(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

# Testing / Validation code
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct

# Copied from MNIST example
def train_all(model, params, train_dataset, test_dataset):

    batch_size = params[0]
    n_epoch = params[1]
    rate_learning = params[2]
    outf = params[3]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    use_cuda = True
    use_mps = False
    if use_cuda & torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps & torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    gamma = 0.7

    optimizer = optim.Adadelta(model.parameters(), lr=rate_learning)
    # optimizer2 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    best_correct = 0
    for epoch in range(1, n_epoch + 1):
        train(model, device, train_loader, optimizer, epoch)
        curr_correct = test(model, device, test_loader)
        scheduler.step()

        # Save model
        if curr_correct > best_correct:
            best_correct = curr_correct
            print("TODO SAVE MODEL, ncorrect =", best_correct)
            # torch.save(model.state_dict(), '%s/cls_model_%d.pth' % (outf, epoch))

    
    cm_train = model_to_cm(model, device, train_loader)
    cm_test = model_to_cm(model, device, test_loader)
    cm_all = cm_train + cm_test
    print("Confusion Matrix:")
    print("Train:")
    print(cm_train)
    print("Test:")
    print(cm_test)
    print("Combined:")
    print(cm_all)

    rs_train = cm_train.diagonal()/cm_train.sum(axis=1)
    rs_test = cm_test.diagonal()/cm_test.sum(axis=1)
    rs_all = cm_all.diagonal()/cm_all.sum(axis=1)
    
    ps_train = cm_train.diagonal()/cm_train.sum(axis=0)
    ps_test = cm_test.diagonal()/cm_test.sum(axis=0)
    ps_all = cm_all.diagonal()/cm_all.sum(axis=0)

    print("Precision Score:")
    print("Train:", ps_train)
    print("test:", ps_test)
    print("Combined:", ps_all)
    
    print("Recall Score:")
    print("Train:", rs_train)
    print("test:", rs_test)
    print("Combined:", rs_all)

# Counter({12: 23478, 11: 9500, 1: 6513, 2: 3603, 0: 3569, 10: 3300, 4: 27, 14: 10})

def model_to_cm(model, device, dataloader):
    """Evaluate the model and then calculate the confusion matrix"""
    model.eval()

    # correct = 0
    pred_all = []
    target_all = []
    with torch.no_grad():
        for data, target in dataloader:
            # data, target = data.to(device), target.to(device)
            data = data.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_all = np.append(pred_all, pred.cpu())
            target_all = np.append(target_all, target)
            # correct += pred.eq(target.view_as(pred)).sum().item()

    cm = confusion_matrix(target_all, pred_all)
    # ps = precision_score(target_all, pred_all)
    # rs = recall_score(target_all, pred_all)

    return cm

def load_and_train():
    print("Starting PyTorch training...")

    print("Loading data...")
    with open(fn, 'rb') as f:
        EventHits, EventTypes = pickle.load(f)

    #batch_size = 128
    batch_size = 400
    epochs = 15
    rate_learning = 0.001
    outf = 'torch_output'

    params = [batch_size, epochs, rate_learning, outf]

    print("Initializing Torch Dataset...")
    
    XBins, YBins, ZBins = 110, 110, 48
    XMin, XMax = -55, 55
    YMin, YMax = -55, 55
    ZMin, ZMax = 0, 48

    dims = [XBins, YBins, ZBins]
    xrange = [XMin, XMax]
    yrange = [YMin, YMax]
    zrange = [ZMin, ZMax]

    ranges = [xrange, yrange, zrange]

    split = 0.8
    
    random.shuffle(shuffledHits)
    random.shuffle(shuffledTypes)

    ceil = math.ceil(len(EventHits)*split)
    EventTypesTrain = shuffledTypes[:ceil]
    EventTypesTest = shuffledTypes[ceil:]
    EventHitsTrain = shuffledHits[:ceil]
    EventHitsTest = shuffledHits[ceil:]

    train_dataset = tm.VoxelDataset(EventHitsTrain, EventTypesTrain, dims, ranges)
    test_dataset = tm.VoxelDataset(EventHitsTest, EventTypesTest, dims, ranges)

    nclasses = train_dataset2.nlabels

    print("Initializing PyTorch version of VoxNet...")
    nclasses = 2
    model = tm.VoxNet(num_classes=nclasses, input_shape=(110, 110, 48))

    time0 = datetime.datetime.now()
    # tm.train(model, params, train_dataset2, test_dataset2)
    tm.train_all(model, params, train_dataset, test_dataset)
    time1 = datetime.datetime.now()
    print("Time to run:", time1-time0)

if __name__ == "__main__":
    voxnet = VoxNet()
    data = torch.rand([256, 1, 32, 32, 32])
    test_out = voxnet(data)

    print("SSS:", test_out)
