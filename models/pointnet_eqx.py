import equinox as eqx
import jax
import jax.numpy as jnp

import optax
# from torch.autograd import Variable
# import torch.nn.functional as F

# import numpy as np

class STN3d(eqx.Module):
    conv1 : eqx.nn.Conv1d
    conv2 : eqx.nn.Conv1d
    conv3 : eqx.nn.Conv1d

    fc1 : eqx.nn.Linear
    fc2 : eqx.nn.Linear
    fc3 : eqx.nn.Linear

    bn1 : eqx.nn.BatchNorm
    bn2 : eqx.nn.BatchNorm
    bn3 : eqx.nn.BatchNorm
    bn4 : eqx.nn.BatchNorm
    bn5 : eqx.nn.BatchNorm

    def __init__(self, channel, key):
        super(STN3d, self).__init__()

        keys = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, padding=1, key=keys[2])
        
        self.fc1 = eqx.nn.Linear(1024, 512, key=keys[3])
        self.fc2 = eqx.nn.Linear(512, 256, key=keys[4])
        self.fc3 = eqx.nn.Linear(256, 9, key=keys[5])
        
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(128, axis_name="batch")
        self.bn3 = eqx.nn.BatchNorm(1024, axis_name="batch")
        self.bn4 = eqx.nn.BatchNorm(512, axis_name="batch")
        self.bn5 = eqx.nn.BatchNorm(256, axis_name="batch")

    def __call__(self, x, state, npts=None):

        # batchsize = x.size()[0]
        D, N = x.size()

        if npts is not None:
            x = x[:, :npts]

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x, state = self.bn3(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn3(self.conv3(x)))

        x = jnp.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x, state = self.bn4(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn4(self.fc1(x)))

        x = self.fc2(x)
        x, state = self.bn5(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)

        # iden = Variable(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)).view(1, 9).repeat(B, 1)
        iden = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32).view(1, 9).repeat(B, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(eqx.Module):
    conv1 : eqx.nn.Conv1d
    conv2 : eqx.nn.Conv1d
    conv3 : eqx.nn.Conv1d

    fc1 : eqx.nn.Linear
    fc2 : eqx.nn.Linear
    fc3 : eqx.nn.Linear

    bn1 : eqx.nn.BatchNorm
    bn2 : eqx.nn.BatchNorm
    bn3 : eqx.nn.BatchNorm
    bn4 : eqx.nn.BatchNorm
    bn5 : eqx.nn.BatchNorm

    k : int

    def __init__(self, k=64, key=None):
        super(STNkd, self).__init__()

        keys = jax.random.split(key, 6)

        self.conv1 = eqx.nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, padding=1, key=keys[2])

        self.fc1 = eqx.nn.Linear(1024, 512, key=keys[3])
        self.fc2 = eqx.nn.Linear(512, 256, key=keys[4])
        self.fc3 = eqx.nn.Linear(256, k * k, key=keys[5])
        
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(128, axis_name="batch")
        self.bn3 = eqx.nn.BatchNorm(1024, axis_name="batch")
        self.bn4 = eqx.nn.BatchNorm(512, axis_name="batch")
        self.bn5 = eqx.nn.BatchNorm(256, axis_name="batch")

        self.k = k

    def __call__(self, x, state, mask=None):
        B, D, N = x.size()
        
        if mask is None:
            mask = jnp.ones([B, N], dtype=jnp.int)

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn1(self.conv1(x)))
        x = x*mask[:, None, :]

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn2(self.conv2(x)))
        x = x*mask[:, None, :]

        x = self.conv3(x)
        x, state = self.bn3(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn3(self.conv3(x)))
        x = x*mask[:, None, :]

        x = jnp.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x, state = self.bn4(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn4(self.fc1(x)))

        x = self.fc2(x)
        x, state = self.bn5(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            # B, 1)
        iden = jnp.eye(self.k).flatten().astype(jnp.float32).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(eqx.Module):
    stn : STN3d

    conv1 : eqx.nn.Conv1d
    conv1b : eqx.nn.Conv1d
    conv2a : eqx.nn.Conv1d
    conv2 : eqx.nn.Conv1d
    conv3 : eqx.nn.Conv1d

    bn1 : eqx.nn.BatchNorm
    bn1b : eqx.nn.BatchNorm
    bn2a : eqx.nn.BatchNorm
    bn2 : eqx.nn.BatchNorm
    bn3 : eqx.nn.BatchNorm

    fstn : STNkd

    global_feat : bool
    feature_transform : bool

    def __init__(self, global_feat=True, feature_transform=True, channel=3, key=None):
        super().__init__()

        keys = jax.random.split(key, 7)

        self.stn = STN3d(channel, keys[5])

        self.conv1 = eqx.nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1, key=keys[0])
        self.conv1b = eqx.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, key=keys[1])
        self.conv2a = eqx.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, key=keys[2])
        self.conv2 = eqx.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, key=keys[3])
        self.conv3 = eqx.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, key=keys[4])
        
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn1b = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn2a = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(128, axis_name="batch")
        self.bn3 = eqx.nn.BatchNorm(1024, axis_name="batch")

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, key=keys[6])

    def __call__(self, x, state, mask=None):
        B, D, N = x.size()

        # if npts is None:
            # npts = N*torch.ones(B, dtype=torch.int)
        
        if mask is None:
            mask = jnp.ones([B, N], dtype=jnp.int)

        # Spatial Transformer: Calculate the rotation matrix and apply it to
        # the x,y,z points while leaving the extra features untransformed
        trans = self.stn(x, mask=mask)

        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        # x = jnp.bmm(x, trans)
        x = x @ trans # jnp.matmul(x, trans)
        if D > 3:
            x = jnp.concatenate([x, feature], axis=2)
        x = x.transpose(2, 1)

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn1(self.conv1(x)))
        x = x*mask[:, None, :]
        # x = zeros_tensors(x, npts)
        
        x = self.conv1b(x)
        x, state = self.bn1b(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn1b(self.conv1b(x)))
        x = x*mask[:, None, :]


        if self.feature_transform:
            trans_feat = self.fstn(x, mask=mask)
            x = x.transpose(2, 1)
            # x = torch.bmm(x, trans_feat)
            x = x @ trans_feat # jnp.matmul(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        x = self.conv2a(x)
        x, state = self.bn2a(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn2a(self.conv2a(x)))
        x = x*mask[:, None, :]
        
        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn2(self.conv2(x)))
        x = x*mask[:, None, :]

        x = self.conv3(x)
        x, state = self.bn3(x, state)
        # x = self.bn3(self.conv3(x))
        x = x*mask[:, None, :]

        # Max pooling???
        x = jnp.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return jnp.concatenate([x, pointfeat], 1), trans, trans_feat

class PointNet(eqx.Module):
    feat : PointNetEncoder
    
    fc1 : eqx.nn.Linear
    fc2 : eqx.nn.Linear
    fc3 : eqx.nn.Linear

    dropout : eqx.nn.Dropout

    bn1 : eqx.nn.BatchNorm
    bn2 : eqx.nn.BatchNorm

    def __init__(self, nclass=1, key=None):
        super().__init__()

        keys = jax.random.split(key, 4)

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=4, key=keys[0])
        self.fc1 = eqx.nn.Linear(1024, 512, key=keys[1])
        self.fc2 = eqx.nn.Linear(512, 256, key=keys[2])
        self.fc3 = eqx.nn.Linear(256, nclass, key=keys[3])

        self.dropout = eqx.nn.Dropout(0.4)

        self.bn1 = eqx.nn.BatchNorm(512, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(256, axis_name="batch")
        
    def __call__(self, x, state, mask=None, key=None):

        x, trans, trans_feat = self.feat(x, mask=mask)

        # Already max pooled over points so mask is now irrelevant

        x = self.fc1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn1(self.fc1(x)))

        x = self.fc2(x)
        x = self.dropout(x, key=key)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        # x = jax.nn.relu(self.bn2(self.dropout(self.fc2(x), key=key)))
        x = self.fc3(x)

        return x, trans_feat
    
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = jnp.eye(d)[None, :, :]
    loss = jnp.mean(jnp.linalg.norm(jnp.matmul(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNetLoss(eqx.Module):
    def __init__(self, loss_fn, mat_diff_loss_scale=0.001):
        super().__init__()
        self.loss_fn = loss_fn
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def __call__(self, logits, target, trans_feat):
        # loss = optax.sigmoid_binary_cross_entropy(logits, target)
        loss = self.loss_fn(logits, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    import numpy as np
    
    key = jax.random.key(42)
    model_key, train_key = jax.random.split(key)

    aaa0 = np.ones([2, 4, 4])
    aaa2 = np.ones([2, 4, 2])

    aaa1 = np.ones_like(aaa0)
    aaa1[1, :, 3:] = 0

    aaa0 = jnp.array(aaa0)
    aaa1 = jnp.array(aaa0)
    aaa2 = jnp.array(aaa0)

    model = PointNet(key=model_key)
    model.eval()

    # model0 = PointNetEncoder(global_feat=True, feature_transform=True, channel=4)
    # model0.eval()

    tmp0, tmp1 = model(aaa0)

    mask = np.ones([2, 4], dtype=jnp.int)
    mask[1, 2:] = 0
    mask = jnp.array(mask)

    tmp2, tmp3 = model(aaa1, mask=mask)
    tmp4, tmp5 = model(aaa2)

    print(tmp0.shape)
    print(tmp1.shape)

    print(tmp0)
    print(tmp2)
    print(tmp4)



