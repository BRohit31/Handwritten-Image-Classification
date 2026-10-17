"""Helpers for the MNIST-subset project (no PyTorch needed to import this module).

Contents
  evaluate, cross_validate, run_all, cv_summary     5-fold CV for sklearn models and neural networks
  sweep, plot_sweep                                  one-hyperparameter CV sweeps
  fit_and_test, count_params                         final test-set evaluation and model size
  HOG, AEFeatures                                    feature extractors usable inside sklearn pipelines
  kmeans_cluster_eval                                unsupervised baseline (clusters mapped to labels)
  run_label_sweep, label_sweep_table, ...            supervised vs. semi-supervised vs. label fraction
  plot_* / top_confusions                            figures for error analysis
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ======================================================================================
# Metrics
# ======================================================================================
def evaluate(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': p, 'macro_recall': r, 'macro_f1': f1}


# ======================================================================================
# Cross-validation
# ======================================================================================
def cross_validate(name, make_model, X, y, kind='sklearn', n_splits=5, seed=42,
                   fit_kwargs=None, inner_val_frac=0.15, verbose=True):
    """Stratified k-fold CV for one model.

    * Same seed => same folds for every model, so results are paired.
    * make_model must return a FRESH unfitted model (nothing carries over between folds).
    * kind='nn': early stopping uses an inner split of the fold's TRAINING part, never the held-out fold.
    """
    X, y = np.asarray(X), np.asarray(y)
    fit_kwargs = dict(fit_kwargs or {})
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows, histories = [], []
    oof_pred = np.full(len(y), -1, dtype=int)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr, X_te, y_te = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
        model = make_model()
        t0 = time.time()
        if kind == 'nn':
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=inner_val_frac, stratify=y_tr, random_state=seed + fold)
            histories.append(model.fit(X_fit, y_fit, X_val, y_val,
                                       seed=seed + fold, verbose=False, **fit_kwargs))
        else:
            model.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        y_pred = np.asarray(model.predict(X_te))
        oof_pred[te_idx] = y_pred
        m = evaluate(y_te, y_pred)
        m.update(fold=fold, fit_time_s=fit_time)
        rows.append(m)
        if verbose:
            print(f"[{name}] fold {fold}/{n_splits}: acc={m['accuracy']:.4f}  "
                  f"macroF1={m['macro_f1']:.4f}  ({fit_time:.1f}s)")
    return {'name': name, 'folds': pd.DataFrame(rows).set_index('fold'),
            'oof_pred': oof_pred, 'y_true': y, 'histories': histories}


def run_all(specs, X, y, n_splits=5, seed=42):
    """specs: name -> dict(make_model=callable, kind='sklearn'|'nn', fit_kwargs=dict)."""
    return {name: cross_validate(name, s['make_model'], X, y, kind=s.get('kind', 'sklearn'),
                                 n_splits=n_splits, seed=seed, fit_kwargs=s.get('fit_kwargs'))
            for name, s in specs.items()}


def cv_summary(results):
    """Table of mean ± std over folds, sorted by mean accuracy (best first)."""
    num = pd.DataFrame({
        n: {'acc': r['folds']['accuracy'].mean(), 'acc_sd': r['folds']['accuracy'].std(),
            'f1': r['folds']['macro_f1'].mean(), 'f1_sd': r['folds']['macro_f1'].std(),
            't': r['folds']['fit_time_s'].mean()} for n, r in results.items()}).T
    num = num.sort_values('acc', ascending=False)
    return pd.DataFrame({'accuracy': [f'{m:.4f} ± {s:.4f}' for m, s in zip(num['acc'], num['acc_sd'])],
                         'macro_f1': [f'{m:.4f} ± {s:.4f}' for m, s in zip(num['f1'], num['f1_sd'])],
                         'fit_time_s': num['t'].round(1)}, index=num.index)


def sweep(make_fn, values, X, y, label, seed=42, n_splits=5):
    """CV accuracy for each hyperparameter value (same folds for all values)."""
    rows = []
    for v in values:
        res = cross_validate(f'{label}={v}', lambda v=v: make_fn(v), X, y, seed=seed,
                             n_splits=n_splits, verbose=False)
        f = res['folds']
        rows.append({'value': v, 'acc_mean': f['accuracy'].mean(), 'acc_std': f['accuracy'].std(),
                     'fit_time_s': f['fit_time_s'].mean()})
        print(f"{label}={v}: acc {rows[-1]['acc_mean']:.4f} ± {rows[-1]['acc_std']:.4f}")
    return pd.DataFrame(rows).set_index('value')


def plot_sweep(ax, df, title, xlabel):
    x = list(range(len(df)))
    ax.errorbar(x, df['acc_mean'], yerr=df['acc_std'], marker='o', capsize=3)
    b = int(np.argmax(df['acc_mean'].values))
    ax.plot(b, df['acc_mean'].iloc[b], 'r*', ms=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in df.index])
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel('CV accuracy'); ax.grid(alpha=0.3)


# ======================================================================================
# Final test-set evaluation and model complexity
# ======================================================================================
def count_params(model):
    """Number of stored numbers defining the model (NN: weights; SVM: support vectors + dual coefs;
    kNN: the stored training set; RF: tree nodes; PCA: components + mean)."""
    if hasattr(model, 'parameters'):                       # torch module or AEFeatures
        return sum(p.numel() for p in model.parameters())
    if isinstance(model, Pipeline):
        return sum(count_params(step) for _, step in model.steps)
    if isinstance(model, LogisticRegression):
        return model.coef_.size + model.intercept_.size
    if isinstance(model, SVC):
        return model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size
    if isinstance(model, KNeighborsClassifier):
        return model.n_samples_fit_ * model.n_features_in_
    if isinstance(model, RandomForestClassifier):
        return sum(e.tree_.node_count for e in model.estimators_)
    if isinstance(model, PCA):
        return model.components_.size + model.mean_.size
    if isinstance(model, HOG):
        return 0
    return np.nan


def fit_and_test(name, spec, x_tr, y_tr, x_te, y_te, seed=42, inner_val_frac=0.15):
    """Refit on the full training set and evaluate ONCE on the test set."""
    model = spec['make_model']()
    t0 = time.time()
    if spec.get('kind') == 'nn':
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_tr, y_tr, test_size=inner_val_frac, stratify=y_tr, random_state=seed)
        hist = model.fit(x_fit, y_fit, x_val, y_val, seed=seed, verbose=False,
                         **spec.get('fit_kwargs', {}))
    else:
        model.fit(x_tr, y_tr)
        hist = None
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = np.asarray(model.predict(x_te))
    pred_time = time.time() - t0
    row = {**evaluate(y_te, y_pred), 'fit_time_s': fit_time,
           'predict_ms_per_img': 1000 * pred_time / len(x_te), 'n_params': count_params(model)}
    return {'row': row, 'model': model, 'y_pred': y_pred, 'history': hist}


# ======================================================================================
# Feature extractors (usable as sklearn pipeline steps)
# ======================================================================================
class HOG(BaseEstimator, TransformerMixin):
    """Histogram of oriented gradients for flattened 28x28 images (stateless)."""

    def __init__(self, pixels_per_cell=4, orientations=9, cells_per_block=2):
        self.pixels_per_cell = pixels_per_cell
        self.orientations = orientations
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            from skimage.feature import hog
        except ImportError as e:
            raise ImportError('HOG features need scikit-image:  pip install scikit-image') from e
        X = np.asarray(X)
        return np.array([hog(img.reshape(28, 28), orientations=self.orientations,
                             pixels_per_cell=(self.pixels_per_cell,) * 2,
                             cells_per_block=(self.cells_per_block,) * 2, block_norm='L2-Hys')
                         for img in X], dtype=np.float32)


class AEFeatures(BaseEstimator, TransformerMixin):
    """Latent codes of a pretrained autoencoder's encoder. encode_fn(encoder, X) -> numpy array."""

    def __init__(self, encoder, encode_fn):
        self.encoder = encoder
        self.encode_fn = encode_fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.encode_fn(self.encoder, X)

    def parameters(self):              # lets count_params() see the encoder weights
        return self.encoder.parameters()


# ======================================================================================
# Unsupervised baseline
# ======================================================================================
def kmeans_cluster_eval(Z_train, y_train, Z_test, y_test, n_clusters=10, seed=42, n_init=10):
    """k-means on Z_train; each cluster gets the majority label of its TRAINING members.
    Reports test accuracy / macro-F1 (needs that label mapping) and NMI / ARI (label-free)."""
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(Z_train)
    mapping = {c: int(np.bincount(y_train[km.labels_ == c], minlength=10).argmax())
               for c in range(n_clusters)}
    clusters_test = km.predict(Z_test)
    y_pred = np.array([mapping[c] for c in clusters_test])
    return {**evaluate(y_test, y_pred),
            'NMI': normalized_mutual_info_score(y_test, clusters_test),
            'ARI': adjusted_rand_score(y_test, clusters_test)}


# ======================================================================================
# Supervised vs. semi-supervised as a function of the number of labels
# ======================================================================================
def stratified_subset(y, frac, seed):
    if frac >= 1.0:
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
    return idx


def run_label_sweep(x_train, y_train, x_test, y_test, label_fracs, seeds, nn_variants, references,
                    fit_kwargs, seed_fn=None, inner_val_frac=0.15):
    """For each label fraction and seed, train every variant on the same labeled subset.

    nn_variants : name -> zero-arg factory returning a fresh NN (fit/predict API of BaseNN)
    references  : name -> zero-arg factory returning a fresh sklearn estimator
    The early-stopping split for the NNs comes from the labeled subset itself (no other labels exist).
    The sklearn references use all labels of the subset.
    """
    rows = []
    for frac in label_fracs:
        for seed in seeds:
            idx = stratified_subset(y_train, frac, seed)
            X_l, y_l = x_train[idx], y_train[idx]
            n_val = max(10, int(round(inner_val_frac * len(idx))))      # at least one validation image per class
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_l, y_l, test_size=n_val, stratify=y_l, random_state=seed)
            base = {'label_frac': frac, 'n_labels': len(idx), 'seed': seed}
            for name, make in nn_variants.items():
                if seed_fn is not None:
                    seed_fn(seed)
                model = make()
                model.fit(X_fit, y_fit, X_val, y_val, seed=seed, verbose=False, **fit_kwargs)
                rows.append({**base, 'method': name, **evaluate(y_test, model.predict(x_test))})
            for name, make in references.items():
                model = make().fit(X_l, y_l)
                rows.append({**base, 'method': name, **evaluate(y_test, model.predict(x_test))})
            print(f'done: {len(idx)} labels ({frac:.0%}), seed {seed}')
    return pd.DataFrame(rows)


def label_sweep_table(df, metric='accuracy'):
    agg = df.groupby(['method', 'n_labels'])[metric].agg(['mean', 'std']).reset_index()
    agg['txt'] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(agg['mean'], agg['std'])]
    return agg.pivot(index='method', columns='n_labels', values='txt')


def plot_label_sweep(df, ax=None, hlines=None):
    """Accuracy vs. number of labels (mean ± std over seeds). hlines: label -> accuracy reference lines."""
    ax = ax or plt.subplots(figsize=(7.5, 4.8))[1]
    for method, g in df.groupby('method'):
        s = g.groupby('label_frac').agg(n=('n_labels', 'first'), mean=('accuracy', 'mean'),
                                        std=('accuracy', 'std'))
        ax.errorbar(s['n'], s['mean'], yerr=s['std'], marker='o', capsize=3, label=method)
    for (label, v), ls in zip((hlines or {}).items(), ['--', ':', '-.']):
        ax.axhline(v, color='gray', ls=ls, lw=1.2, label=label)
    ax.set_xscale('log'); ax.set_xlabel('number of labeled training images (log scale)')
    ax.set_ylabel('test accuracy'); ax.set_title('Accuracy vs. amount of labeled data')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return ax


def paired_gain(df, method_a, method_b, metric='accuracy'):
    """Paired difference (a - b) per label fraction; compare the mean with the std across seeds."""
    piv = df.pivot_table(index=['label_frac', 'seed'], columns='method', values=metric)
    diff = (piv[method_a] - piv[method_b]).rename(f'{method_a} - {method_b}')
    return diff.groupby('label_frac').agg(['mean', 'std']).round(4)


# ======================================================================================
# Figures for error analysis
# ======================================================================================
def plot_nn_curves(results, names=None):
    """Mean ± std train/validation accuracy per epoch over the CV folds, for NN models."""
    names = names or [n for n, r in results.items() if r['histories']]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.6), squeeze=False)
    for ax, n in zip(axes[0], names):
        for key, label in [('train_acc', 'train'), ('val_acc', 'validation')]:
            hs = [h[key] for h in results[n]['histories']]
            L = max(len(h) for h in hs)
            M = np.array([h + [h[-1]] * (L - len(h)) for h in hs])   # early stopping => unequal lengths
            ax.plot(range(1, L + 1), M.mean(0), label=label)
            ax.fill_between(range(1, L + 1), M.mean(0) - M.std(0), M.mean(0) + M.std(0), alpha=0.2)
        ax.set_title(n); ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title, ax, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, fontsize=10)
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f'{v:.2f}' if normalize else int(v), ha='center', va='center',
                        fontsize=6, color='white' if v > 0.5 else 'black')


def top_confusions(y_true, y_pred, k=6):
    cm = confusion_matrix(y_true, y_pred)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((i, j, off[i, j]) for i in range(10) for j in range(10) if off[i, j] > 0),
                   key=lambda t: -t[2])[:k]
    return pd.DataFrame([{'true': i, 'predicted': j, 'count': int(c),
                          '% of true class': round(100 * c / cm[i].sum(), 1)} for i, j, c in pairs],
                        columns=['true', 'predicted', 'count', '% of true class'])


def plot_misclassified(x, y_true, y_pred, pairs, n=8, title=''):
    if len(pairs) == 0:
        print('No misclassified images to show.')
        return
    fig, axes = plt.subplots(len(pairs), n, figsize=(1.5 * n, 1.8 * len(pairs)), squeeze=False)
    for r, (t, p) in enumerate(pairs):
        idx = np.where((y_true == t) & (y_pred == p))[0][:n]
        for c in range(n):
            axes[r, c].axis('off')
            if c < len(idx):
                axes[r, c].imshow(x[idx[c]].reshape(28, 28), cmap='gray')
                axes[r, c].set_title(f'{t}→{p}', fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
