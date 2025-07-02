import torch
import torch.nn.functional as F

import numpy as np

class STN3d(torch.nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9)
        
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

    def forward(self, x, mask=None):

        # batchsize = x.size()[0]
        B, D, N = x.size()
        
        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int).to(x.device)

        # if npts is None:
            # npts = x.size()[2]*torch.ones(batchsize, dtype=torch.int)

        x = self.relu(self.bn1(self.conv1(x)))
        # x = zeros_tensors(x, npts)
        x = x*mask[:, None, :]

        x = self.relu(self.bn2(self.conv2(x)))
        # x = zeros_tensors(x, npts)
        x = x*mask[:, None, :]
        x = self.relu(self.bn3(self.conv3(x)))
        # x = zeros_tensors(x, npts)
        x = x*mask[:, None, :]

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)).view(1, 9).repeat(B, 1)
        iden = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32).view(1, 9).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(torch.nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k * k)
        
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, mask=None):
        B, D, N = x.size()
        
        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int).to(x.device)

        x = self.relu(self.bn1(self.conv1(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            # B, 1)
        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(1, self.k * self.k).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(torch.nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        super().__init__()
        self.stn = STN3d(channel)

        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv1b = torch.nn.Conv1d(64, 64, 1)
        self.conv2a = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn1b = torch.nn.BatchNorm1d(64)
        self.bn2a = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, mask=None):
        B, D, N = x.size()

        # if npts is None:
            # npts = N*torch.ones(B, dtype=torch.int)
        
        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int).to(x.device)

        # Spatial Transformer: Calculate the rotation matrix and apply it to
        # the x,y,z points while leaving the extra features untransformed
        trans = self.stn(x, mask=mask)

        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)


        x = F.relu(self.bn1(self.conv1(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)
        
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = x*mask[:, None, :]


        if self.feature_transform:
            trans_feat = self.fstn(x, mask=mask)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = x*mask[:, None, :]
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)
        x = self.bn3(self.conv3(x))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)

        # Max pooling???
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(torch.nn.Module):
    def __init__(self, nclass=1):
        super().__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=4)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, nclass)

        self.dropout = torch.nn.Dropout(0.4)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x, mask=None):

        x, trans, trans_feat = self.feat(x, mask=mask)

        # Already max pooled over points so mask is now irrelevant

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x, trans_feat
    
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNetLoss(torch.nn.Module):
    def __init__(self, loss_fn, mat_diff_loss_scale=0.001):
        super().__init__()
        self.loss_fn = loss_fn
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, logits, target, trans_feat):
        # loss = F.binary_cross_entropy_with_logits(logits, target)
        loss = self.loss_fn(logits, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

'''
class PointNetLoss2(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, logits, target, trans_feat):
        loss = F.cross_entropy(logits, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
'''

if __name__ == '__main__':
    aaa0 = torch.ones([2, 4, 4])
    aaa2 = torch.ones([2, 4, 2])

    aaa1 = torch.ones_like(aaa0)
    aaa1[1, :, 3:] = 0

    torch.manual_seed(0)

    model = PointNet()
    model.eval()

    model0 = PointNetEncoder(global_feat=True, feature_transform=True, channel=4)
    model0.eval()

    tmp0, tmp1 = model(aaa0)

    mask = torch.ones([2, 4], dtype=torch.int)
    mask[1, 2:] = 0

    tmp2, tmp3 = model(aaa1, mask=mask)
    tmp4, tmp5 = model(aaa2)

    print(tmp0.shape)
    print(tmp1.shape)

    print(tmp0)
    print(tmp2)
    print(tmp4)



