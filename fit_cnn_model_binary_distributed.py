import pickle
import random
import datetime
import argparse
import os

# import numpy as np

import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

def blue(x): return '\033[94m' + x + '\033[0m'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class VoxelDataset(Dataset):
    def __init__(self, data, labels, dims, ranges):
        self.data = data  # x, y, z, e for each hit in each event
        self.labels = labels  # type label number for each event

        # labels_unique = np.unique(labels)
        labels_unique = torch.unique(torch.Tensor(labels))

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

        # self.xbins2 = np.linspace(self.xmin, self.xmax, num=self.xbins+1)
        # self.ybins2 = np.linspace(self.ymin, self.ymax, num=self.ybins+1)
        # self.zbins2 = np.linspace(self.zmin, self.zmax, num=self.zbins+1)

        self.xbins2 = torch.linspace(self.xmin, self.xmax, steps=self.xbins+1)
        self.ybins2 = torch.linspace(self.ymin, self.ymax, steps=self.ybins+1)
        self.zbins2 = torch.linspace(self.zmin, self.zmax, steps=self.zbins+1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Generate the tensor for this idx"""

        # Get the hits and the label for this event
        data_idx = self.data[idx]
        label_idx = self.labels[idx]

        tensor = torch.zeros([1, self.xbins, self.ybins, self.zbins])
        # nhits = len(data_idx)

        # Subtract 1 since digitize will also give indices for values
        # outside the input bins
        # xbin2 = np.digitize(data_idx[:, 0], self.xbins2) - 1
        # ybin2 = np.digitize(data_idx[:, 1], self.ybins2) - 1
        # zbin2 = np.digitize(data_idx[:, 2], self.zbins2) - 1

        # Subtract 1 since bucketize will also give indices for values
        # outside the input bins and index 0 is for values below the lowest
        # bin
        xbin2 = torch.bucketize(torch.Tensor(data_idx[:, 0]), self.xbins2) - 1
        ybin2 = torch.bucketize(torch.Tensor(data_idx[:, 1]), self.ybins2) - 1
        zbin2 = torch.bucketize(torch.Tensor(data_idx[:, 2]), self.zbins2) - 1

        energy = data_idx[:, 3]        

        for xbin, ybin, zbin, eval in zip(xbin2, ybin2, zbin2, energy):
            tensor[0, xbin, ybin, zbin] += eval

        # Iterate over all the hits, find the bin for each hit and add its energy to
        # that bin
        # for i in data_idx:
            # xbin = (int)(((i[0] - self.xmin) / (self.xmax - self.xmin)) * self.xbins)
            # ybin = (int)(((i[1] - self.ymin) / (self.ymax - self.ymin)) * self.ybins)
            # zbin = (int)(((i[2] - self.zmin) / (self.zmax - self.zmin)) * self.zbins)
            # tensor[0, xbin, ybin, zbin] += i[3]
        
        # Separating into Pair (10-19) and Compton (0-9)
        # if label_idx >= 10:
            # label_out = 1
        # else:
            # label_out = 0
        label_out = label_idx*1.0

        # return tensor, self.label_map[label_idx]
        return tensor, label_out

class TestNet1(torch.nn.Module):

    def __init__(self, input_shape=(110, 110, 48)):
        super(TestNet1, self).__init__()
        
        # Convolutional Layers
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=5)
        
        self.relu = torch.nn.ReLU()
        self.leakyrelu0p01 = torch.nn.LeakyReLU(0.01)
        self.leakyrelu0p1 = torch.nn.LeakyReLU(0.1)

        self.dropout0p5 = torch.nn.Dropout(0.5)
        self.dropout0p2 = torch.nn.Dropout(0.2)
        self.dropout3d0p5 = torch.nn.Dropout3d(0.5)

        self.conv_list = [self.conv1, self.relu, 
                          self.conv2, self.relu, self.dropout3d0p5]
                        #   self.conv3, self.relu, self.dropout3d0p5]
                        #   self.conv4, self.relu, self.dropout3d0p5]
        
        x = self.cnnpart(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = x.size()[1:].numel()
        print("Number of features:", first_fc_in_features)

        self.flatten = torch.nn.Flatten()
        self.lfc1 = torch.nn.Linear(first_fc_in_features, 128)
        self.lfc2 = torch.nn.Linear(128, 1)

        self.conn_list = [self.flatten,
                          self.lfc1, self.relu, self.dropout0p5,
                          self.lfc2]


    def cnnpart(self, x):
        # for f in self.conv_list:
            # x = f(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d0p5(x)

        return x
    
    def fullyconnectedpart(self, x):
        # for f in self.conn_list:
            # x = f(x)

        x = self.flatten(x)
        x = self.lfc1(x)
        x = self.relu(x)
        x = self.dropout0p5(x)
        x = self.lfc2(x)

        return x

    def forward(self, x):
        x = self.cnnpart(x)
        x = self.fullyconnectedpart(x)
        return x

class ComPairNet(torch.nn.Module):

    def __init__(self, input_shape=(32, 32, 32)):
        """
        ComPairNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.

        Modified in order to accept different input shapes.

        Parameters
        ----------
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        """

        super(ComPairNet, self).__init__()
        self.body = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv3d(in_channels=1,
                                      out_channels=32, kernel_size=5, stride=2)),
            # ('lkrelu1', torch.nn.LeakyReLU(0.1)),
            ('relu1', torch.nn.ReLU()),
            # ('drop1', torch.nn.Dropout(p=dropout_val)),
            ('drop1', torch.nn.Dropout3d(p=0.5)),
            ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            # ('lkrelu2', torch.nn.LeakyReLU(0.1)),
            ('relu2', torch.nn.ReLU()),
            # ('drop2', torch.nn.Dropout(p=dropout_val))
            ('drop2', torch.nn.Dropout3d(p=0.5))
        ]))

        # Trick to accept different input shapes. Just puts a random
        # variable through the body and checks its output dimensions
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = x.size()[1:].numel()
        # first_fc_in_features = 1
        # for n in x.size()[1:]:
            # first_fc_in_features *= n

        print("Number of features:", first_fc_in_features)

        self.head = torch.nn.Sequential(OrderedDict([
            ('flat', torch.nn.Flatten()),
            ('fc1', torch.nn.Linear(first_fc_in_features, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.5)),
            ('fc2', torch.nn.Linear(128, 1)),
        ]))

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x

'''
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
'''

# Training code in the MNIST example
def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10, dry_run=False):
    model.train()

    loss_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("TEST:", output.shape, target.shape)
        loss = loss_fn(output, target.reshape((-1, 1)))
        loss_train += loss.item()*len(target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

    loss_train /= len(train_loader.dataset)

    return loss_train

# Testing / Validation code
def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.view_as(output)).item()*len(target)

            pred = output.ge(0).type(torch.int)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct, test_loss

# Copied from MNIST example
def train_all(model, params, train_dataset, test_dataset, dir='./', label=""):

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


    # optimizer = optim.Adadelta(model.parameters(), lr=rate_learning)
    optimizer = optim.Adam(model.parameters(), lr=rate_learning)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Every 10 epochs multiply the learning rate by a factor of gamma
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    best_correct = 0
    loss_train = []
    loss_test = []
    correct_test = []
    for epoch in range(1, n_epoch + 1):

        # Train and then append the avg of the average losses for each batch
        avg_loss_train = train(model, device, train_loader, optimizer, epoch, loss_fn)
        loss_train.append(avg_loss_train)

        # Run the test set and store the number of correct and the overall average loss
        curr_correct, avg_loss = test(model, device, test_loader, loss_fn)
        loss_test.append(avg_loss)
        correct_test.append(curr_correct)

        scheduler.step()

        # Save model
        if curr_correct > best_correct:
            best_correct = curr_correct
            print("Saving new model, ncorrect =", best_correct)
            fn_state = "test_torch_model_params_" + label + ".pth"
            fn_state = os.path.join(dir, fn_state)
            # torch.save(model.module.state_dict(), "test_torch_model_params_May14.pth")
            torch.save(model.module.state_dict(), fn_state)


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

    print("Loss Train:", loss_train)
    print("Loss Validation:", loss_test)
    print("Correct Validation", correct_test)

    fn_loss = "loss_acc_values_" + label + ".txt"
    fn_loss = os.path.join(dir, fn_loss)
    with open(fn_loss, "w") as f:
        for list_tmp in (loss_train, loss_test, correct_test):
            for tmp in list_tmp:
                f.write(str(tmp))
                f.write(" ")
            f.write("\n")

# Counter({12: 23478, 11: 9500, 1: 6513, 2: 3603, 0: 3569, 10: 3300, 4: 27, 14: 10})

def model_to_cm(model, device, dataloader):
    """Evaluate the model and then calculate the confusion matrix"""
    model.eval()

    pred_all = torch.Tensor([])
    target_all = torch.Tensor([])
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = output.ge(0).type(torch.int)
            pred_all = torch.cat((pred_all, pred.cpu()))
            target_all = torch.cat((target_all, target))

    cm = confusion_matrix(target_all, pred_all)

    return cm

def load_and_train(fn, dir="./", label="", modelname='ComPairNet', batch_size=800):
    print("Starting PyTorch training...")

    print("Loading data...", fn)
    # fn = '/data/slag2/njmille2/test_dataset_nhits12.pkl'
    with open(fn, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    print(f"Dataset contains {len(event_hits)} events")

    print(f"Batch size = {batch_size}")

    # batch_size = 128
    # batch_size = 800
    epochs = 20
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

    zipped = list(zip(event_hits, event_types))
    random.shuffle(zipped)
    shuffledHits, shuffledTypes = zip(*zipped)

    # random.shuffle(event_hits)
    # random.shuffle(event_types)

    split = 0.9
    # ceil = np.ceil(len(event_hits)*split).astype(int)
    ceil = int(len(event_hits)*split)
    EventTypesTrain = shuffledTypes[:ceil]
    EventTypesTest = shuffledTypes[ceil:]
    EventHitsTrain = shuffledHits[:ceil]
    EventHitsTest = shuffledHits[ceil:]

    print(f"Train: {len(EventTypesTrain)}, Test: {len(EventTypesTest)}")

    train_dataset = VoxelDataset(EventHitsTrain, EventTypesTrain, dims, ranges)
    test_dataset = VoxelDataset(EventHitsTest, EventTypesTest, dims, ranges)

    if modelname == 'ComPairNet':
        print("Initializing PyTorch binary classification version of ComPairNet...")
        model = ComPairNet(input_shape=(XBins, YBins, ZBins))
    elif modelname == 'TestNet1':
        print("Initializing PyTorch binary classification version of TestNet1...")
        model = TestNet1(input_shape=(XBins, YBins, ZBins))
    else:
        raise ValueError("Bad modelname")

    time0 = datetime.datetime.now()
    train_all(model, params, train_dataset, test_dataset, dir=dir, label=label)
    time1 = datetime.datetime.now()
    print("Time to run:", time1-time0)

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
    
    args = parser.parse_args()
    
    load_and_train(args.fn, dir=args.dir, label=args.label, modelname=args.model, batch_size=args.batch)