"""PyTorch models for the MNIST-subset project.

  BaseNN                       mini-batch training with early stopping, batched predict / predict_proba
  MLP, CNN                     supervised networks (flat 784-dim input)
  Autoencoder + train_autoencoder, encode, reconstruct     unsupervised stage
  SemiSupervisedClassifier     encoder (pretrained or random) + classification head
"""
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _as_tensor(a, dtype):
    if isinstance(a, torch.Tensor):
        return a.to(dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype)


# ======================================================================================
# Base class with mini-batch training
# ======================================================================================
class BaseNN(nn.Module):
    """Mini-batch training, early stopping on a validation set, batched inference."""

    def __init__(self, device=None):
        super().__init__()
        self.device = device or get_default_device()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = None                 # created by subclasses once the parameters exist

    def forward(self, X):
        raise NotImplementedError

    @torch.no_grad()
    def _logits(self, X, batch_size=1024):
        self.eval()
        X = _as_tensor(X, torch.float32)
        return torch.cat([self.forward(X[i:i + batch_size].to(self.device)).cpu()
                          for i in range(0, len(X), batch_size)])

    def predict(self, X, batch_size=1024):
        return self._logits(X, batch_size).argmax(-1).numpy()

    def predict_proba(self, X, batch_size=1024):
        return torch.softmax(self._logits(X, batch_size), dim=-1).numpy()

    def fit(self, X, Y, X_val=None, Y_val=None, epochs=50, batch_size=64,
            patience=5, seed=None, verbose=True):
        """Every epoch: reshuffle, split into mini-batches, one optimizer step per batch.
        With a validation set: keep the weights of the best validation accuracy and stop after
        `patience` epochs without improvement. Returns per-epoch train/val loss and accuracy."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        X = _as_tensor(X, torch.float32)
        Y = _as_tensor(Y, torch.long)
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else 0)
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True, generator=g,
                            drop_last=(len(X) % batch_size == 1))   # BatchNorm cannot train on a batch of 1

        use_val = X_val is not None and Y_val is not None
        if use_val:
            Xv = _as_tensor(X_val, torch.float32)
            Yv = _as_tensor(Y_val, torch.long)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_acc, best_state, wait = -1.0, None, 0

        for epoch in range(epochs):
            self.train()
            running_loss, correct, n = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.forward(xb)
                loss = self.loss(logits, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * len(xb)
                correct += (logits.argmax(-1) == yb).sum().item()
                n += len(xb)
            history['train_loss'].append(running_loss / n)
            history['train_acc'].append(correct / n)
            msg = (f"epoch {epoch + 1:3d}/{epochs}  loss {history['train_loss'][-1]:.4f}  "
                   f"acc {history['train_acc'][-1]:.4f}")

            if use_val:
                logits_v = self._logits(Xv)
                v_loss = self.loss(logits_v, Yv).item()
                v_acc = (logits_v.argmax(-1) == Yv).float().mean().item()
                history['val_loss'].append(v_loss)
                history['val_acc'].append(v_acc)
                msg += f"  | val loss {v_loss:.4f}  val acc {v_acc:.4f}"
                if v_acc > best_acc:
                    best_acc, wait = v_acc, 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(msg)
                            print(f'Early stopping at epoch {epoch + 1} (best val acc {best_acc:.4f})')
                        break
            if verbose:
                print(msg)

        if best_state is not None:
            self.load_state_dict(best_state)
        return history


# ======================================================================================
# Supervised networks
# ======================================================================================
class MLP(BaseNN):
    """784 -> 512 -> 256 -> 128 -> 10 with dropout."""

    def __init__(self, hidden=(512, 256, 128), p=0.3, lr=1e-3, weight_decay=1e-4, device=None):
        super().__init__(device)
        layers, d_in = [], 784
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p)]
            d_in = h
        layers.append(nn.Linear(d_in, 10))
        self.layers = nn.Sequential(*layers)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, X):
        return self.layers(X)


class CNN(BaseNN):
    """Two conv blocks + dense head. Takes flat 784-dim input and reshapes it to (N, 1, 28, 28)."""

    def __init__(self, p=0.3, lr=1e-3, weight_decay=1e-4, device=None):
        super().__init__(device)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(p), nn.Linear(128, 10),
        )
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, X):
        return self.classifier(self.features(X.view(-1, 1, 28, 28)))


# ======================================================================================
# Unsupervised stage: autoencoder
# ======================================================================================
class DeepEncoder(nn.Module):
    def __init__(self, latent_dim=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DeepDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),               # pixels are scaled to [0, 1]
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = DeepEncoder(latent_dim)
        self.decoder = DeepDecoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(x_unlabeled, latent_dim=64, epochs=25, batch_size=64, lr=1e-3,
                      weight_decay=1e-5, seed=42, verbose=True):
    """Train on images only (no labels). Returns (autoencoder in eval mode, per-epoch MSE list)."""
    device = get_default_device()
    torch.manual_seed(seed)
    ae = Autoencoder(latent_dim).to(device)
    X = torch.as_tensor(x_unlabeled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True,
                        drop_last=(len(X) % batch_size == 1))
    opt = optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        ae.train()
        total = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            loss = crit(ae(bx), bx)
            loss.backward()
            opt.step()
            total += loss.item() * len(bx)
        history.append(total / len(X))
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print(f'epoch {epoch + 1:2d}/{epochs}  reconstruction MSE {history[-1]:.5f}')
    return ae.eval(), history


@torch.no_grad()
def encode(encoder, X, batch_size=1024):
    """Latent codes (numpy) for images X."""
    encoder.eval()
    dev = next(encoder.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([encoder(X[i:i + batch_size].to(dev)).cpu()
                      for i in range(0, len(X), batch_size)]).numpy()


@torch.no_grad()
def reconstruct(ae, X, batch_size=1024):
    """Autoencoder reconstructions (numpy) of images X."""
    ae.eval()
    dev = next(ae.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([ae(X[i:i + batch_size].to(dev)).cpu()
                      for i in range(0, len(X), batch_size)]).numpy()


# ======================================================================================
# Stage 2: classifier on top of an encoder
# ======================================================================================
class SemiSupervisedClassifier(BaseNN):
    """encoder (pretrained or randomly initialised) + classification head, same API as the other NNs."""

    def __init__(self, encoder, latent_dim=64, freeze_encoder=False, p=0.3,
                 lr=1e-3, weight_decay=1e-4, device=None):
        super().__init__(device)
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for prm in self.encoder.parameters():
                prm.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 10),
        )
        self.to(self.device)
        self.optimizer = optim.Adam([q for q in self.parameters() if q.requires_grad],
                                    lr=lr, weight_decay=weight_decay)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:        # frozen means frozen: no BatchNorm statistic updates, no dropout
            self.encoder.eval()
        return self

    def forward(self, X):
        return self.head(self.encoder(X))
