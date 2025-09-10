import torch

from torch.utils.data import Dataset

def get_amegox_xyzbin2(xpos, ypos, zpos, xbins2, ybins2):
    zvals = [-45.5,   -43.9,   -42.3,   -40.7,   -36.432, -34.932, -33.432, -31.932, -30.432,
             -28.932, -27.432, -25.932, -24.432, -22.932, -21.432, -19.932, -18.432, -16.932,
             -15.432, -13.932, -12.432, -10.932,  -9.432,  -7.932,  -6.432,  -4.932,  -3.432,
             -1.932,  -0.432,   1.068,   2.568,   4.068,   5.568,   7.068,   8.568,  10.068,
             11.568,  13.068,  14.568,  16.068,  17.568,  19.068,  20.568,  22.068]
    zvals = torch.tensor(zvals)
    
    zvals2 = zvals + 0.5

    zbins = torch.searchsorted(zvals2, torch.tensor(zpos))


    # Original bucketize
    xbins = torch.bucketize(torch.Tensor(xpos), xbins2) - 1
    ybins = torch.bucketize(torch.Tensor(ypos), ybins2) - 1

    return xbins, ybins, zbins

#Get the bin idx for the discrete z positions in AMEGO-X
def get_amegox_xyzbin(xpos, ypos, zpos):
    zvals = [-45.5,   -43.9,   -42.3,   -40.7,   -36.432, -34.932, -33.432, -31.932, -30.432,
             -28.932, -27.432, -25.932, -24.432, -22.932, -21.432, -19.932, -18.432, -16.932,
             -15.432, -13.932, -12.432, -10.932,  -9.432,  -7.932,  -6.432,  -4.932,  -3.432,
             -1.932,  -0.432,   1.068,   2.568,   4.068,   5.568,   7.068,   8.568,  10.068,
             11.568,  13.068,  14.568,  16.068,  17.568,  19.068,  20.568,  22.068]
    zvals = torch.tensor(zvals)
    
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
    bins = torch.linspace(xymin, xymax, steps=nbins+1)

    xbins = torch.bucketize(torch.tensor(xpos), bins) - 1
    ybins = torch.bucketize(torch.tensor(ypos), bins) - 1

    # Shift by a small amount so values around the unique value should be put into the correct bin even
    # if for some reason there is some small error in the value
    zvals2 = zvals + 0.5

    zbins = torch.searchsorted(zvals2, torch.tensor(zpos))

    return xbins, ybins, zbins

class AMEGOXVoxelDataset(Dataset):
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
        self.zbins = 44

        xrange = ranges[0]
        yrange = ranges[1]

        self.xmin = xrange[0]
        self.xmax = xrange[1]
        self.ymin = yrange[0]
        self.ymax = yrange[1]

        self.xbins2 = torch.linspace(self.xmin, self.xmax, steps=self.xbins+1)
        self.ybins2 = torch.linspace(self.ymin, self.ymax, steps=self.ybins+1)

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
        xbin2, ybin2, zbin2 = get_amegox_xyzbin2(data_idx[:, 0], data_idx[:, 1], data_idx[:, 2],
                                                 self.xbins2, self.ybins2)
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

class AMEGOXPointCloud(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        labels_unique = torch.unique(torch.Tensor(labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data_idx = torch.tensor(self.data[idx])
        label_idx = torch.tensor(self.labels[idx])

        return data_idx, label_idx

def pc_collate_fn(batch):

    nbatch = len(batch)

    npts = 0
    for data, _ in batch:
        npts_tmp = len(data)
        if npts_tmp > npts:
            npts = npts_tmp

    data_out = torch.zeros([nbatch, 4, npts])
    mask_out = torch.zeros([nbatch, npts])
    label_out = torch.zeros(nbatch)
    for idx, data in enumerate(batch):
        pts = data[0]
        label = data[1]
        npts_tmp = len(pts)
        data_out[idx, :, :npts_tmp] = pts.T
        label_out[idx] = label
        mask_out[idx, :npts_tmp] = 1

    # return data_out, mask_out, label_out
    return data_out, label_out
