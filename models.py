import torch

from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    def __init__(self, data, labels, dims, ranges, extra=False):
        self.data = data  # x, y, z, e for each hit in each event
        self.labels = labels  # type label number for each event

        self.extra = extra

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

        # Subtract 1 since bucketize will also give indices for values
        # outside the input bins and index 0 is for values below the lowest
        # bin
        xbin2 = torch.bucketize(torch.Tensor(data_idx[:, 0]), self.xbins2) - 1
        ybin2 = torch.bucketize(torch.Tensor(data_idx[:, 1]), self.ybins2) - 1
        zbin2 = torch.bucketize(torch.Tensor(data_idx[:, 2]), self.zbins2) - 1

        energy = data_idx[:, 3]        

        for xbin, ybin, zbin, eval in zip(xbin2, ybin2, zbin2, energy):
            tensor[0, xbin, ybin, zbin] += eval

        label_out = label_idx*1.0

        label_out = 1.0
        if self.extra:
            return tensor, label_out, len(data_idx)
        else:
            return tensor, label_out

def gen_testnet1(input_shape=(110, 110, 48)):

    # Convolutional layers
    conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
    conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
    conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5)
    conv4 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)

    # Dropout layers
    dropout0p4 = torch.nn.Dropout(0.4)
    # dropout0p2 = torch.nn.Dropout(0.2)
    # dropout3d0p4 = torch.nn.Dropout3d(0.4)
    
    # ReLU layers
    leakyrelu0p01 = torch.nn.LeakyReLU(0.01)
    # relu = torch.nn.ReLU()

    conv_list = [conv1, leakyrelu0p01, dropout0p4,
                 conv2, leakyrelu0p01, dropout0p4,
                 conv3, leakyrelu0p01, dropout0p4,
                ]

    x = torch.rand((1, 1) + input_shape)
    for f in conv_list:
        x = f(x)

    # x = self.cnnpart(torch.autograd.Variable(
            # torch.rand((1, 1) + input_shape)))
    
    first_fc_in_features = x.size()[1:].numel()
    print("NUM FEATURES:", first_fc_in_features)
    
    flatten = torch.nn.Flatten()

    # Linear layers
    lfc1 = torch.nn.Linear(first_fc_in_features, 128)
    lfc2 = torch.nn.Linear(128, 1)

    linear_list = [flatten,
                   lfc1, leakyrelu0p01, dropout0p4,
                   lfc2,
                  ]

    # return conv_list + linear_list
    all_layers = conv_list + linear_list
    return torch.nn.Sequential(*all_layers)

def gen_testnet_mult(input_shape=(110, 110, 48)):
    """
    Returns a similar model to gen_testnet1 except that there are three output
    categories instead of one.
    """    
    # Convolutional layers
    conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
    conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
    conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5)
    conv4 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)

    # Dropout layers
    dropout0p4 = torch.nn.Dropout(0.4)
    # dropout0p2 = torch.nn.Dropout(0.2)
    # dropout3d0p4 = torch.nn.Dropout3d(0.4)
    
    # ReLU layers
    leakyrelu0p01 = torch.nn.LeakyReLU(0.01)
    # relu = torch.nn.ReLU()

    conv_list = [conv1, leakyrelu0p01, dropout0p4,
                 conv2, leakyrelu0p01, dropout0p4,
                 conv3, leakyrelu0p01, dropout0p4,
                 conv4, leakyrelu0p01, dropout0p4,
                ]

    x = torch.rand((1, 1) + input_shape)
    for f in conv_list:
        x = f(x)

    # x = self.cnnpart(torch.autograd.Variable(
            # torch.rand((1, 1) + input_shape)))
    
    first_fc_in_features = x.size()[1:].numel()
    
    flatten = torch.nn.Flatten()

    # Linear layers
    lfc1 = torch.nn.Linear(first_fc_in_features, 128)
    lfc2 = torch.nn.Linear(128, 3)

    linear_list = [flatten,
                   lfc1, leakyrelu0p01, dropout0p4,
                   lfc2,
                  ]

    # return conv_list + linear_list
    all_layers = conv_list + linear_list
    return torch.nn.Sequential(*all_layers)

def gen_claudenet1(input_shape=(110, 110, 48)):

    relu = torch.nn.ReLU()

    conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    
    bn1 = torch.nn.BatchNorm3d(32)
    bn2 = torch.nn.BatchNorm3d(64)
    bn3 = torch.nn.BatchNorm3d(128)
    
    pool1 = torch.nn.MaxPool3d(kernel_size=2)
    pool2 = torch.nn.MaxPool3d(kernel_size=2)
    pool3 = torch.nn.MaxPool3d(kernel_size=2)

    dropout0p5 = torch.nn.Dropout(0.5)
    
    conv_list = [conv1, bn1, relu, pool1,
                 conv2, bn2, relu, pool2,
                 conv3, bn3, relu, pool3,
                ]
    
    x = torch.rand((1, 1) + input_shape)
    for f in conv_list:
        x = f(x)
    
    first_fc_in_features = x.size()[1:].numel()
    
    flatten = torch.nn.Flatten()
    
    flat_features = 128 * 13 * 13 * 6 # for 110x110x48 input

    print("FEATURES:", first_fc_in_features, flat_features)

    fc1 = torch.nn.Linear(first_fc_in_features, 512)
    fc2 = torch.nn.Linear(512, 256)
    fc3 = torch.nn.Linear(256, 1)
    
    linear_list = [flatten,
                   fc1, relu, dropout0p5,
                   fc2, relu, dropout0p5,
                   fc3,
                  ]
    
    all_layers = conv_list + linear_list
    return torch.nn.Sequential(*all_layers)


