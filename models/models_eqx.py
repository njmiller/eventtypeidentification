import equinox as eqx
import jax
import jax.numpy as jnp

from torch.utils.data import Dataset

def get_amegox_xyzbin2(xpos, ypos, zpos, xbins2, ybins2):
    zvals = [-45.5,   -43.9,   -42.3,   -40.7,   -36.432, -34.932, -33.432, -31.932, -30.432,
             -28.932, -27.432, -25.932, -24.432, -22.932, -21.432, -19.932, -18.432, -16.932,
             -15.432, -13.932, -12.432, -10.932,  -9.432,  -7.932,  -6.432,  -4.932,  -3.432,
             -1.932,  -0.432,   1.068,   2.568,   4.068,   5.568,   7.068,   8.568,  10.068,
             11.568,  13.068,  14.568,  16.068,  17.568,  19.068,  20.568,  22.068]
    zvals = jnp.array(zvals)
    
    zvals2 = zvals + 0.5

    zbins = jnp.searchsorted(zvals2, jnp.array(zpos))


    # Original bucketize
    xbins = jnp.bucketize(jnp.array(xpos), xbins2) - 1
    ybins = jnp.bucketize(jnp.array(ypos), ybins2) - 1

    return xbins, ybins, zbins

#Get the bin idx for the discrete z positions in AMEGO-X
def get_amegox_xyzbin(xpos, ypos, zpos):
    zvals = [-45.5,   -43.9,   -42.3,   -40.7,   -36.432, -34.932, -33.432, -31.932, -30.432,
             -28.932, -27.432, -25.932, -24.432, -22.932, -21.432, -19.932, -18.432, -16.932,
             -15.432, -13.932, -12.432, -10.932,  -9.432,  -7.932,  -6.432,  -4.932,  -3.432,
             -1.932,  -0.432,   1.068,   2.568,   4.068,   5.568,   7.068,   8.568,  10.068,
             11.568,  13.068,  14.568,  16.068,  17.568,  19.068,  20.568,  22.068]
    zvals = jnp.array(zvals)
    
    xpos0, ypos0 = xpos[0], ypos[0]

    #Towers are from 0 to 44 with sign
    if xpos0 < 0:
        xpos = -xpos

    if ypos0 < 0:
        ypos = -ypos

    # Drop positions that now have a negative value
    ypos = ypos[xpos > 0]
    zpos = zpos[xpos > 0]
    xpos = xpos[xpos > 0]

    xpos = xpos[ypos > 0]
    zpos = zpos[ypos > 0]
    ypos = ypos[ypos > 0]

    # idx = xpos > 0
    # xpos, ypos, zpos = xpos[idx], ypos[idx], zpos[idx]
    # idx = ypos > 0
    # xpos, ypos, zpos = xpos[idx], ypos[idx], zpos[idx]
    
    # 110 bins from 0 to 44. X/Y positions seem to run from around -43.75 to 43.75
    nbins = 110
    xymin = 0
    xymax = 44
    bins = jnp.linspace(xymin, xymax, steps=nbins+1)

    xbins = jnp.bucketize(jnp.array(xpos), bins) - 1
    ybins = jnp.bucketize(jnp.array(ypos), bins) - 1

    # Shift by a small amount so values around the unique value should be put into the correct bin even
    # if for some reason there is some small error in the value
    zvals2 = zvals + 0.5

    zbins = jnp.searchsorted(zvals2, jnp.array(zpos))

    return xbins, ybins, zbins

class VoxelDataset(Dataset):
    def __init__(self, data, labels, dims, ranges, extra=False, experiment=None):
        self.data = data  # x, y, z, e for each hit in each event
        self.labels = labels  # type label number for each event
        self.experiment = experiment

        self.extra = extra

        # labels_unique = np.unique(labels)
        labels_unique = jnp.unique(jnp.array(labels))

        self.nlabels = len(labels_unique)
        self.label_map = {}
        for i, val in enumerate(labels_unique):
            self.label_map[val] = i

        # Info related to converting the point cloud to a voxel tensor
        self.dims = dims
        self.xbins = dims[0]
        self.ybins = dims[1]
        self.zbins = dims[2]

        if self.experiment == "AMEGOX":
            print("Setting AMEGO-X to 44 z bins.")
            self.zbins = 44

        xrange = ranges[0]
        yrange = ranges[1]
        zrange = ranges[2]

        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.ymin = yrange[0]
        self.ymax = yrange[1]
        self.zmin = zrange[0]
        self.zmax = zrange[1]

        self.xbins2 = jnp.linspace(self.xmin, self.xmax, steps=self.xbins+1)
        self.ybins2 = jnp.linspace(self.ymin, self.ymax, steps=self.ybins+1)
        self.zbins2 = jnp.linspace(self.zmin, self.zmax, steps=self.zbins+1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Generate the tensor for this idx"""

        # Get the hits and the label for this event
        data_idx = self.data[idx]
        label_idx = self.labels[idx]

        tensor = jnp.zeros([1, self.xbins, self.ybins, self.zbins])

        # Subtract 1 since bucketize will also give indices for values
        # outside the input bins and index 0 is for values below the lowest
        # bin
        if self.experiment == "AMEGOX":
            # xbin2, ybin2, zbin2 = get_amegox_xyzbin(data_idx[:, 0], data_idx[:, 1], data_idx[:, 2])
            xbin2, ybin2, zbin2 = get_amegox_xyzbin2(data_idx[:, 0], data_idx[:, 1], data_idx[:, 2],
                                                     self.xbins2, self.ybins2)
        else:
            xbin2 = jnp.bucketize(jnp.array(data_idx[:, 0]), self.xbins2) - 1
            ybin2 = jnp.bucketize(jnp.array(data_idx[:, 1]), self.ybins2) - 1
            zbin2 = jnp.bucketize(jnp.array(data_idx[:, 2]), self.zbins2) - 1

        energy = data_idx[:, 3]        

        for xbin, ybin, zbin, eval in zip(xbin2, ybin2, zbin2, energy):
            tensor[0, xbin, ybin, zbin] += eval

        label_out = label_idx*1.0
        # label_out = 1.0*(label_idx // 10)
        # print("TEST:", label_idx, label_out)

        # label_out = 1.0
        if self.extra:
            return tensor, label_out, len(data_idx)
        else:
            return tensor, label_out

def gen_claudenet1(input_shape=(110, 110, 48), key=None):
    keys = jax.random.split(key, 7)

    relu = jax.nn.relu

    conv1 = eqx.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1, key=keys[0])
    conv2 = eqx.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, key=keys[1])
    conv3 = eqx.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1, key=keys[2])
    conv4 = eqx.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1, key=keys[3])
    
    bn1 = eqx.nn.BatchNorm(32, axis_name="batch")
    bn2 = eqx.nn.BatchNorm(64, axis_name="batch")
    bn3 = eqx.nn.BatchNorm(128, axis_name="batch")
    bn4 = eqx.nn.BatchNorm(256, axis_name="batch")

    pool1 = eqx.nn.MaxPool3d(kernel_size=2)
    pool2 = eqx.nn.MaxPool3d(kernel_size=2)
    pool3 = eqx.nn.MaxPool3d(kernel_size=2)
    pool4 = eqx.nn.MaxPool3d(kernel_size=2)

    dropout0p5 = eqx.nn.Dropout(0.5)
    
    conv_list = [conv1, bn1, relu, pool1,
                 conv2, bn2, relu, pool2,
                 conv3, bn3, relu, pool3,
                #  conv4, bn4, relu, pool4,
                ]
    
    key = jax.random.key(seed=42)
    x = jax.random.normal(key, shape=(1, 1) + input_shape)
    for f in conv_list:
        x = f(x)
    
    first_fc_in_features = x.size()[1:].numel()
    
    flatten = eqx.nn.Flatten()
    
    flat_features = 128 * 13 * 13 * 6 # for 110x110x48 input

    print("FEATURES:", first_fc_in_features, flat_features)

    fc1 = eqx.nn.Linear(first_fc_in_features, 512, key=keys[4])
    fc2 = eqx.nn.Linear(512, 256, key=keys[5])
    fc3 = eqx.nn.Linear(256, 1, key=keys[6])
    
    linear_list = [flatten,
                   fc1, relu, dropout0p5,
                   fc2, relu, dropout0p5,
                   fc3,
                  ]
    
    all_layers = conv_list + linear_list
    # return eqx.nn.Sequential(*all_layers)
    return all_layers


