from itertools import product

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve


def _prep(normal_scores: np.ndarray, anomaly_scores: np.ndarray):
    y_true = np.hstack([np.zeros_like(normal_scores), np.ones_like(anomaly_scores)])
    y_pred = np.hstack([normal_scores, anomaly_scores])

    return y_true, y_pred


def _auroc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return auc(fpr, tpr), fpr, tpr


def auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray):
    """Computes and returns the Area under Receiver operating characteristic.

    Args:
        normal_scores:
            scores for each normal image, a single score
        anomaly_scores
            scores for anomaly data

    """
    return _auroc(*_prep(normal_scores, anomaly_scores))


def one_vs_one_scores(normal_scores, anomaly_scores, tpr_target=0.95, verbose=False):
    """Computes the metrics auroc, aupr, and fprn (for the given tpr_target)"""
    y_true, y_pred = _prep(normal_scores, anomaly_scores)
    auroc, fpr, tpr = _auroc(y_true, y_pred)

    ret_val = {
        'auroc': auroc,
        'aupr': average_precision_score(y_true, y_pred),
        f"fpr{int(tpr_target * 100)}": fpr[np.argmax(tpr >= tpr_target)],
    }

    if verbose:
        ret_val['roc'] = {
            'fpr': fpr,
            'tpr': tpr
        }

    return ret_val


def one_vs_one_scores_over_classifiers(classifiers, validation_output,  out_of_distribution_data, perturbed_data):
    model_evaluations = [(cls_name, dataset_name, one_vs_one_scores(cls(validation_output), cls(model_output)))
                         for (cls_name, cls), (dataset_name, model_output)
                         in (product(classifiers, [*out_of_distribution_data, *perturbed_data]))]

    _t = pd.DataFrame(model_evaluations, columns=['Classifier', 'Dataset', 'metrics'])
    _t = _t.drop('metrics', axis=1).join(pd.DataFrame(_t['metrics'].values.tolist()))
    _tt = _t.groupby(['Classifier'])[['auroc', 'aupr', 'fpr95']].mean().reset_index()
    _tt['Dataset'] = 'Overall'

    return pd.concat((_tt, _t)).sort_values(['Classifier', 'Dataset'])[_t.columns]


def kullback_leibler_divergence_to_multivariate_standard_normal(xs):
    """Calculates the Kullback-Leibler Divergence between the given samples to a multivariate standard Gaussian
    distribution with mean of 0 and standard deviation of 1.

    Note that the input features are expected to be in column form. Meaning each row in `xs` is a sample.
    """
    if len(xs.shape) > 2:
        xs = xs.reshape(xs.shape[0], -1)

    xs_mean = xs.T.mean(1)
    xs_var = xs.T.var(1)

    if isinstance(xs, torch.Tensor):
        _log_fn = torch.log
        _sum_fn = torch.sum
    else:
        _log_fn = np.log
        _sum_fn = np.sum

    return 0.5 * (_sum_fn(xs_var + xs_mean ** 2 - _log_fn(xs_var)) - xs.shape[1])
