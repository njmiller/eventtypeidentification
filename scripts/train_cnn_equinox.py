import pickle
import argparse
from functools import partial
import os
import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax.numpy as jnp
import jax
import jax.sharding as jshard
import jax.experimental.mesh_utils as mesh_utils
from torch.utils.data import DataLoader, random_split
import torch
import optax

# TODO: Copy model to EquinoxCNN and copy VoxelDataset code here
from data.datasets import AMEGOXVoxelDataset
    
devices = jax.local_devices()
n_devices = jax.local_device_count()

devices_mesh = mesh_utils.create_device_mesh((n_devices, ))
sharding = jshard.PositionalSharding(devices_mesh)
replicated = sharding.replicate()

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
STEPS = 20
PRINT_EVERY = 30
SEED = 5678
NUM_EPOCHS = 20

key = jax.random.PRNGKey(SEED)

class EquinoxCNN(eqx.Module):
    conv1 : eqx.nn.Conv3d
    conv2 : eqx.nn.Conv3d
    conv3 : eqx.nn.Conv3d
    bn1 : eqx.nn.BatchNorm
    bn2 : eqx.nn.BatchNorm
    bn3 : eqx.nn.BatchNorm
    pool : eqx.nn.Pool
    dropout : eqx.nn.Dropout
    fc1 : eqx.nn.Linear
    fc2 : eqx.nn.Linear
    fc3 : eqx.nn.Linear

    def __init__(self, nbins, key):
        super().__init__()

        keys = jax.random.split(key, 7)

        self.conv1 = eqx.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1, key=keys[2])
        # self.conv4 = eqx.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1, key=keys[3])
    
        self.bn1 = eqx.nn.BatchNorm(32, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(64, axis_name="batch")
        self.bn3 = eqx.nn.BatchNorm(128, axis_name="batch")
        # self.bn4 = eqx.nn.BatchNorm(256, axis_name="batch")

        self.pool = eqx.nn.MaxPool3d(kernel_size=2, stride=2)

        self.dropout = eqx.nn.Dropout(0.5)
    
        # first_fc_in_features = x.size()[1:].numel()
    
        zbins = 44
        xbins = nbins[0]
        ybins = nbins[1]

        xbins_end = xbins // 2 // 2 // 2
        ybins_end = ybins // 2 // 2 // 2
        zbins_end = zbins // 2 // 2 // 2

        flat_features = 128 * xbins_end * ybins_end * zbins_end # for 110x110x48 input

        print("FLAT FEATURES:", flat_features)

        # print("FEATURES:", first_fc_in_features, flat_features)

        # self.fc1 = eqx.nn.Linear(first_fc_in_features, 512, key=keys[4])
        self.fc1 = eqx.nn.Linear(flat_features, 512, key=keys[4])
        self.fc2 = eqx.nn.Linear(512, 256, key=keys[5])
        self.fc3 = eqx.nn.Linear(256, 1, key=keys[6])

    def __call__(self, x, state, key):
        # if key is None:
            # training = False
        # else:
            # training = True
            # keys = jax.random.split(key, 2)

        if key is not None:
            keys = jax.random.split(key, 2)

        if state is None:
            raise ValueError("State is None")
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.pool(x)
        
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        x = self.pool(x)
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x, state = self.bn3(x, state)
        x = jax.nn.relu(x)
        x = self.pool(x)

        # Flatten and fully connected layers
        # x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.ravel(x)


        x = self.fc1(x)
        x = jax.nn.relu(x)
        if key is not None:
            x = self.dropout(x, key=keys[0])

        x = self.fc2(x)
        x = jax.nn.relu(x)
        if key is not None:
            x = self.dropout(x, key=keys[1])

        x = self.fc3(x)
        
        return x, state

# Don't need to JIT all these functions because they are only called inside the JITed train_step
# and eval_step

def torch_to_jax(batch):
    """Convert PyTorch tensors to JAX arrays."""
    volumes, labels = batch
    volumes = jnp.array(volumes.numpy())
    labels = jnp.array(labels.numpy())
    return volumes, labels

# Define a loss function that we can use Jax for the gradient in order to backpropagate errors
def loss_fn(model, state, x, y, key):
    
    dropout_key, _ = jax.random.split(key)

    model = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))

    logits_y, new_state = model(x, state, dropout_key)

    loss_tmp = optax.sigmoid_binary_cross_entropy(logits_y, y)
    return jnp.mean(loss_tmp), (new_state, logits_y)

def accuracy(logits, labels):
    """Compute accuracy."""
    predictions = (logits >= 0).astype(int)

    acc = jnp.mean(predictions == labels)

    return acc

def confusion_matrix(logits, labels):
    """Compute the 2x2 confusion matrix."""

    # TODO: Figure out how to generate an NxN matrix given the number of unique labels

    y_pred = (logits >= 0).astype(int)
    y_true = labels

    tp = jnp.sum((y_true == 1) & (y_pred == 1))
    tn = jnp.sum((y_true == 0) & (y_pred == 0))
    fp = jnp.sum((y_true == 0) & (y_pred == 1))
    fn = jnp.sum((y_true == 1) & (y_pred == 0))
    
    return jnp.array([[tn, fp], [fn, tp]])

@eqx.filter_jit
def train_step(model, state, opt_state, volumes, labels, key, optimizer):
    # Compute loss and gradients
    (loss, (new_state, logits)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, volumes, labels, key)

    # Update model parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Compute accuracy
    acc = accuracy(logits, labels)

    return new_model, new_state, new_opt_state, loss, acc
    
    # return train_step

@eqx.filter_jit(donate="all")
def parallel_train_step(model, state, opt_state, x, y, key, optimizer, sharding):
    replicated = sharding.replicate()

    model, state, opt_state = eqx.filter_shard((model, state, opt_state), replicated)
    sharding_x = sharding.reshape((-1, 1, 1, 1, 1))

    x = eqx.filter_shard(x, sharding_x)
    y = eqx.filter_shard(y, sharding)

    # Compute loss and gradients
    (loss, (state, logits)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, x, y, key)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    acc = accuracy(logits, y)

    model, state, opt_state = eqx.filter_shard((model, state, opt_state), replicated)

    return model, state, opt_state, loss, acc

@eqx.filter_jit
def eval_step(model, state, volumes, labels):
# def eval_step(model, volumes, labels, state):
    # logits, new_state = forward_pass(model, state, volumes, labels)

    model = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))
    logits, _ = model(volumes, state, None)

    # Full confusion matrix instead of accuracy during eval step
    # acc = accuracy(logits, labels)
    cm = confusion_matrix(logits, labels)

    return cm
    
    # return eval_step

@eqx.filter_jit(donate="all-except-first")
def parallel_eval_step(model, state, x, y, sharding):
    replicated = sharding.replicate()
    model, state = eqx.filter_shard((model, state), replicated)
    
    sharding_x = sharding.reshape((-1, 1, 1, 1, 1))

    x = eqx.filter_shard(x, sharding_x)
    y = eqx.filter_shard(y, sharding)
    
    model = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))
    logits, _ = model(x, y, None)

    cm = confusion_matrix(logits, y)

    return cm

def get_loaders(fn, nxy, rangexy, batch_size):
    """Given the filename for AMEGO-X data and info regarding the X/Y pixelization, generate
    the PyTorch DataLoaders."""

    dims = [nxy, nxy]
    xrange = [rangexy[0], rangexy[1]]
    yrange = [rangexy[0], rangexy[1]]

    ranges = [xrange, yrange]

    # Create datasets and dataloaders
    with open(fn, 'rb') as f:
        event_hits, event_types = pickle.load(f)

    dataset_all = AMEGOXVoxelDataset(event_hits, event_types, dims, ranges, extra=False)

    split = 0.9 
    ntrain = int(len(dataset_all)*split)
    nval = len(dataset_all) - ntrain

    train_dataset, val_dataset = random_split(dataset_all, [ntrain, nval],
            generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Binary Classification of Events')

    parser.add_argument('-fn', dest='fn', action='store', help='Dataset filename')
    parser.add_argument('-label', dest='label', action='store', default="",
                        help='Label to add to output data')
    parser.add_argument('-dir', dest='dir', action='store', default="./",
                        help='Directory for output data')
    parser.add_argument("-batch", dest='batch', action='store', type=int, default=800,
                        help="Batch size")
    parser.add_argument("-nxy", dest='nxy', action='store', type=int, default=110,
                        help="Number of bins in X/Y direction")
    parser.add_argument("-rangexy", dest='rangexy', action='store', nargs='+', type=int, default=[-55, 55],
                        help="Range for bins (Single value instead of list has the code figure out the limits)")
    
    args = parser.parse_args()

    key = jax.random.key(42)
    model_key, train_key = jax.random.split(key)

    # Load the data and generate the PyTorch DataLoaders
    train_loader, val_loader = get_loaders(args.fn, args.nxy, args.rangexy, args.batch)

    # Initialize model
    nbins = [args.nxy, args.nxy]
    model, state = eqx.nn.make_with_state(EquinoxCNN)(nbins=nbins, key=model_key)

    # Initilize optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    (model, state, opt_state) = eqx.filter_shard((model, state, opt_state), replicated)

    best_correct = 0

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        total_train_loss = 0.0
        total_train_acc = 0.0
        num_train_batches = 0

        time0 = datetime.datetime.now()
        for batch_idx, batch in enumerate(train_loader):
            volumes, labels = torch_to_jax(batch)

            # Split key for this batch
            train_key, batch_key = jax.random.split(train_key)

            # Perform training step
            model, state, opt_state, loss, acc = parallel_train_step(
                model, state, opt_state, volumes, labels, batch_key, optimizer, sharding
            )

            time1 = datetime.datetime.now()
            time_elapsed = time1 - time0
            percent_finished = 100. * batch_idx / len(train_loader)
            if percent_finished > 0:
                time_to_finish = time_elapsed * (100 - percent_finished) / percent_finished
            else:
                time_to_finish = time_elapsed * 10000

            total_train_loss += loss
            total_train_acc += acc
            num_train_batches += 1

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, "
                      f"Time: {time_elapsed} : {time_to_finish}, "
                      f"Loss: {loss:.4f}, Acc: {acc:.4f}", end='\r')
                
        # Validation phase
        # total_val_acc = 0.0
        total_val_cm = jnp.zeros([2, 2])

        inference_model = eqx.nn.inference_mode(model)
        for batch in val_loader:
            volumes, labels = torch_to_jax(batch)

            # Perform evaluation step
            cm = parallel_eval_step(inference_model, state, volumes, labels, sharding)

            total_val_cm += cm
            # total_val_acc += acc
            num_val_batches += 1

        # Print epoch statistics
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_acc = total_train_acc /  num_train_batches
        total_correct = total_val_cm[0, 0] + total_val_cm[1, 1]
        avg_val_acc = total_correct / jnp.sum(total_val_cm)

        # Precision and Recall
        tp = total_val_cm[0, 0]
        tn = total_val_cm[1, 1]
        fp = total_val_cm[0, 1]
        fn = total_val_cm[1, 0]

        prec_n = tn / (tn + fn)
        prec_p = tp / (tp + fp)
        prec = [prec_n, prec_p]

        rec_p = tp / (tp + fn)
        rec_n = tn / (tn + fp)

        rec = [rec_n, rec_p]
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_acc:.4f}, "
              f"Val Acc: {avg_val_acc:.4f}")
        print("-" * 60)

        # If best accuracy then save model to file
        if total_correct > best_correct:
            best_correct = total_correct
            print("Saving new model, ncorrect =", best_correct)
            fn_state = "test_eqx_model_params_" + args.label + ".pkl"
            fn_state = os.path.join(args.dir, fn_state)
            with open(fn_state, "wb") as f:
                eqx.tree_serialise_leaves(f, model)


    print("Training completed!")

if __name__ == '__main__':
    main()

