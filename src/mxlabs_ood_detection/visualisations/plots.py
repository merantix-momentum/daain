from itertools import chain
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from research_projects.ood_detection.data.perturbed_dataset import PerturbedImage
from sklearn.metrics import auc, roc_curve

from mxlabs_ood_detection.data import PerturbedDataset


def plot_perturbed_image(img: PerturbedImage, figsize: Tuple[int, int] = None, title_suffix=None):
    """Returns the plot for the given input image, using all attacks present.

    Args:
        img: PerturbedImage
            The image to plot.
        figsize: Tuple
            If you want to overwrite the default `figsize`, should be adapted to the number of attacks.
    """
    # fig, axs = plt.subplots(ncols=4, nrows=len(img.perturbed) + 1, figsize=figsize or (20, 3 * len(img.perturbed)))
    fig, axs = plt.subplots(
        ncols=4, nrows=len(img["perturbed"]) + 1, figsize=figsize or (20, 3 * len(img["perturbed"]))
    )

    normal_input = np.array(img["normal_image"][0])
    normal_output = np.array(img["normal_output"][0])

    axs[0][0].imshow(normal_input)
    axs[0][0].set_title("Input")

    axs[0][1].imshow(np.ones_like(normal_input) * 255)
    axs[0][1].set_title("Abs. diff to normal input")

    axs[0][2].imshow(normal_output)
    axs[0][2].set_title("Mask")

    axs[0][3].imshow(np.ones_like(normal_output) * 255)
    axs[0][3].set_title("Abs. diff to normal output")

    for i, (attack, attacked_image) in enumerate(img["perturbed"].items()):
        axs[i + 1][0].set_ylabel(attack)
        axs[i + 1][0].imshow(attacked_image["perturbed_image"][0])
        axs[i + 1][1].imshow(np.abs(normal_input - np.array(attacked_image["perturbed_image"][0])))
        axs[i + 1][2].imshow(attacked_image["output"][0])
        axs[i + 1][3].imshow(np.abs(normal_output - np.array(attacked_image["output"][0])))

    for axs_ in axs:
        for ax in axs_:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{title_suffix}\nOverview of attacks on image {img.image_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    return fig


def plot_tensor(x: torch.Tensor, numpy_only: bool = False) -> object:
    if len(x.shape) == 3:
        # Assuming that 3 channel tensors only need to switch axis
        # CHW -> HWC

        x = x.permute(1, 2, 0) * 255
    img_ndarray = x.byte().cpu().numpy()
    if numpy_only:
        return img_ndarray
    return Image.fromarray(img_ndarray)


def plot_coactivations(
    zarr_root, attacks, log_transform=False, image_id="total", title_part="features of pixel", labels="pixel"
):
    fig, axs = plt.subplots(ncols=len(attacks), nrows=2, figsize=(5 * len(attacks), 15))

    for i, attack in enumerate(attacks):
        if i == 0:
            zero_image_placeholder = np.zeros_like(zarr_root[f"unperturbed/{image_id}"])
            if log_transform:
                cs = axs[0][i].imshow(np.log(np.array(zarr_root[f"unperturbed/{image_id}"]) * 10 ** 5 + 0.001))
            else:
                cs = axs[0][i].imshow(zarr_root[f"unperturbed/{image_id}"])
            axs[0][i].set_title("unperturbed")
            axs[0][i].set_xlabel(labels)
            axs[0][i].set_ylabel(labels)
        else:
            axs[0][i].imshow(zero_image_placeholder)

        if log_transform:
            axs[1][i].imshow(np.log(np.array(zarr_root[f"{attack}/{image_id}"]) * 10 ** 5 + 0.001))
        else:
            axs[1][i].imshow(zarr_root[f"{attack}/{image_id}"])

        axs[1][i].set_title(attack)
        axs[1][i].set_xlabel(labels)
        axs[1][i].set_ylabel(labels)
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])

    fig.colorbar(cs, ax=axs, orientation="horizontal", fraction=0.1)
    fig.suptitle(
        f"Comparison of co-activation over {title_part} (every 20th pixel)"
        f"{'(log transformed)' if log_transform else ''}",
        fontsize="x-large",
    )
    # fig.tight_layout()

    return fig


def non_super_large_vals(vals, median):
    # third_quantile = np.percentile(vals, 75)
    # return vals[vals <= third_quantile]
    return vals[vals <= 2 * median]


def discarded_stats(vals, median):
    non_discared = len(non_super_large_vals(vals, median))
    return f"left over: {non_discared}/{len(vals)}"


def plot_score_comparison(normal_scores, attacked_scores, experiment_id):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    colors = sns.color_palette()

    gen_plot_args = {"kde": True, "stat": "probability", "bins": 100}

    plot_args = {
        "normal": {"data": normal_scores, "label": "normal", "color": colors[0], **gen_plot_args},
        "attacked": {"data": attacked_scores, "label": "attacked", "color": colors[1], **gen_plot_args},
    }

    axs[0][0].set_title("Over all scores")
    sns.histplot(ax=axs[0][0], **plot_args["normal"])
    sns.histplot(ax=axs[0][0], **plot_args["attacked"])

    median_attacked_scores = np.median(attacked_scores)
    median_normal_scores = np.median(normal_scores)

    # print(f"median: attacked: {median_attacked_scores}, normal: {median_normal_scores}")
    # print(f"normal: {discarded_stats(normal_scores, median_normal_scores)}")
    # print(f"attacked: {discarded_stats(attacked_scores, median_attacked_scores)}")

    axs[0][1].set_title("Over only scores lower than 2 * median")
    sns.histplot(
        ax=axs[0][1], **{**plot_args["normal"], "data": non_super_large_vals(normal_scores, median_normal_scores)}
    )
    sns.histplot(
        ax=axs[0][1], **{**plot_args["attacked"], "data": non_super_large_vals(attacked_scores, median_attacked_scores)}
    )

    # sns.histplot(non_super_large_vals(normal_scores, median_normal_scores),
    #              label='normal', ax=axs[1], color=colors[0], **plot_args)
    # sns.histplot(non_super_large_vals(attacked_scores, median_attacked_scores),
    #              label='attacked', ax=axs[1], color=colors[1], **plot_args)

    axs[1][0].set_title("Only normal ")
    axs[1][1].set_title("Only attacked")
    sns.histplot(
        ax=axs[1][0], **{**plot_args["normal"], "data": non_super_large_vals(normal_scores, median_normal_scores)}
    )
    sns.histplot(
        ax=axs[1][1], **{**plot_args["attacked"], "data": non_super_large_vals(attacked_scores, median_attacked_scores)}
    )
    # sns.histplot(ax=axs[1][0], **plot_args['normal'])
    # sns.histplot(ax=axs[1][1], **plot_args['attacked'])

    plt.suptitle(f"Histogram over the anomaly scores for model\n{experiment_id}")
    axs[0][1].legend()

    return fig


def plot_histogram(Y_train, Y_val, Ys_perturbed, Ys_ood, title):
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
    colors = sns.color_palette(n_colors=2)
    sns.histplot(Y_train, label="trainer", stat="probability", ax=axs[0], bins=50, color=colors[0])
    sns.histplot(Y_val, label="val", stat="probability", ax=axs[0], bins=50, color=colors[1])

    colors = sns.color_palette(n_colors=len(Ys_perturbed) + len(Ys_ood))

    for i, (name, Y_perturbed) in enumerate(Ys_perturbed + Ys_ood):
        sns.histplot(Y_perturbed, label=name, stat="probability", ax=axs[1], color=colors[i], bins=40)

    fig.suptitle(f"Histogram over anomaly scores\n{title}")

    for a in axs:
        a.legend()

    return fig


def plot_roc_results(results):
    def _get_plot_data(normal_scores, attacked_scores, ood_scores):
        y_true = np.hstack([np.zeros_like(normal_scores), np.ones_like(attacked_scores), np.ones_like(ood_scores)])
        y_pred = np.hstack([normal_scores, attacked_scores, ood_scores])

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    fig = plt.figure(figsize=(15, 15))

    for name, (fpr, tpr, roc_auc) in (
        (k, _get_plot_data(r["Y_val"], r["Y_perturbed"], r["Y_ood"]))
        for k, r in sorted(results, key=lambda x: x[1]["roc_area"], reverse=True)
    ):
        plt.plot(fpr, tpr, label=f"{name} (area = {roc_auc:0.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristics")
    plt.legend(loc="lower right")

    return fig


def _plot_loss_curve_(results, experiment_id=None, ax=None):
    def _inner_(r, val):
        t = np.array(r[val])
        return np.vstack([t, np.full_like(t, r["run_id"]), np.arange(t.shape[0], dtype=int)])

    val_losses = pd.DataFrame(
        np.hstack([_inner_(r, "val_losses") for r in results]).T, columns=["val_loss", "run_id", "epoch"]
    )

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    # TODO this should be equal to len(results) ...?
    colors = sns.color_palette(n_colors=val_losses["run_id"].nunique() + 1)
    sns.lineplot(data=val_losses, x="epoch", y="val_loss", ax=ax, label="aggregated", color=colors[0], linewidth=2)
    sns.lineplot(data=val_losses, x="epoch", y="val_loss", ax=ax, hue="run_id", palette=colors[1:], linestyle="--")
    if experiment_id is not None:
        ax.set_title(f"Validation loss over multiple runs\n{experiment_id}")
    else:
        ax.set_title("Validation loss over multiple runs")


def _color_palette_(results):
    r = results[0]
    keys = list(r["Ys_perturbed"].keys()) + list(r["Ys_ood"].keys()) + ["overall"]

    return {n: c for n, c in zip(keys, sns.color_palette(n_colors=len(keys)))}


def _aggregate_roc_curves(roc_curves: List[Tuple[List, List]]) -> pd.DataFrame:
    """Aggregates and 'aligns' roc-curves, useful if you want to plot them over multiple runs.
    """

    fprs = np.linspace(0, 1, num=100)

    return pd.DataFrame(
        chain(
            *[
                zip(np.full_like(fprs, i + 1), fprs, np.interp(fprs, roc_curves[i][0], roc_curves[i][1]))
                for i in range(len(roc_curves))
            ]
        ),
        columns=["run_id", "fpr", "tpr"],
    )


#
# For the plots below: Note the parameter `ax`, which is used to combine multiple plots.
#


def _plot_roc_curves_over_runs_(results, colors, experiment_id=None, ax=None):
    aggregated_overall_roc = _aggregate_roc_curves([(r["overall_fpr"], r["overall_tpr"]) for r in results])
    mean_overall_auc = np.mean([r["roc_area"] for r in results])
    kl_scenarios = (
        pd.DataFrame([{**r["kl_divergences"]["perturbed"], **r["kl_divergences"]["ood"]} for r in results])
        .mean()
        .to_dict()
    )

    # looping over the colors to preserve the order
    aggregated_data_roc = {
        dt: (
            _aggregate_roc_curves([(r["fpr_tpr_pairs"][dt]["fpr"], r["fpr_tpr_pairs"][dt]["tpr"]) for r in results]),
            np.mean([r["fpr_tpr_pairs"][dt]["roc"] for r in results]),
            kl_scenarios[dt],
        )
        for dt in colors.keys()
        if dt != "overall"
    }

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    sns.lineplot(
        data=aggregated_overall_roc,
        x="fpr",
        y="tpr",
        color=colors["overall"],
        linewidth=3,
        ax=ax,
        label=f"Overall (AUC: {mean_overall_auc:0.2f})",
    )

    for k, (agged, mean_auc, mean_kl) in aggregated_data_roc.items():
        sns.lineplot(
            data=agged,
            x="fpr",
            y="tpr",
            color=colors[k],
            ax=ax,
            label=f"{k} (mAUC: {mean_auc:0.2f}, mKL: {mean_kl:0.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    if experiment_id is not None:
        ax.set_title(f"Aggregated ROC over multiple runs\n{experiment_id}")
    else:
        ax.set_title("Aggregated ROC over multiple runs")

    ax.legend(loc="lower right")


def _plot_histogram(result, colors, axs=None, n_bins=50):
    if axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)

    _colors = sns.color_palette(n_colors=2)
    sns.histplot(
        result["Y_train"],
        label=f"trainer (KL: {result['kl_divergences']['trainer']:0.2f})",
        stat="probability",
        ax=axs[0],
        bins=n_bins,
        color=_colors[0],
    )
    sns.histplot(
        result["Y_val"],
        label=f"val (KL: {result['kl_divergences']['val']:0.2f})",
        stat="probability",
        ax=axs[0],
        bins=n_bins,
        color=_colors[1],
    )

    scenarios = {**result["Ys_perturbed"], **result["Ys_ood"]}
    kl_scenarios = {**result["kl_divergences"]["perturbed"], **result["kl_divergences"]["ood"]}

    for n, c in colors.items():
        if n != "overall":
            sns.histplot(
                scenarios[n],
                label=f"{n} (KL {kl_scenarios[n]:0.2f})",
                stat="probability",
                ax=axs[1],
                color=c,
                bins=n_bins,
            )

    # fig.suptitle(f"Histogram over anomaly scores\n{title}")

    for a in axs:
        a.legend()
        a.set_xlim([0.0, 1.0])
        a.set_ylim([0.0, 1.0])
        a.set_xlabel("Anomaly Score")

    axs[0].set_title(f"Train vs Val (AUC: {result['roc_area']:0.2f})")
    axs[1].set_title("Scenarios")


def plot_run_results(results, path=None):
    fig = plt.figure(constrained_layout=True, figsize=(30, 15))

    gs = GridSpec(3, 4, figure=fig)

    ax_loss = fig.add_subplot(gs[0, :2])
    ax_roc = fig.add_subplot(gs[0:, 2:])
    ax_hists = [
        (fig.add_subplot(gs[1, :1]), fig.add_subplot(gs[2, :1])),
        (fig.add_subplot(gs[1, 1:2]), fig.add_subplot(gs[2, 1:2])),
    ]

    colors = _color_palette_(results)
    _plot_loss_curve_(results, ax=ax_loss)
    _plot_roc_curves_over_runs_(results, colors, ax=ax_roc)

    sorted_results = sorted(results, key=lambda x: x["roc_area"], reverse=True)
    for r, axs in zip(sorted_results[:2], ax_hists):
        _plot_histogram(r, colors, axs=axs)

    fig.suptitle(f"{sorted_results[0]['evaluation_string']}\nn_runs: {len(results)}")

    if path is not None:
        fig.savefig(path)


# TODO refactor this method
# def plot_outliers(results, attacks, out_of_distributions, root_path_to_attacked_images):
#    best_run_result = max(results, key=lambda x: x['roc_area'])
#
#    data = PerturbedDataset(attacks=[*attacks, *out_of_distributions], root_path=root_path_to_attacked_images)
#    unperturbed_dataset = AD.ActivationDataset(zarr_root=T._get_zarr_root(zarr_group=zarr_group, mask=mask),
#                                               transforms=transforms)
#
#    normal_scores = best_run_result['Y_val']
#    attacked_scores = best_run_result['Ys_perturbed']['FGSM_Iter_Target_Person']
#
#    P.plot_perturbed_image(
#        data.get_image_by_id(unperturbed_dataset.keys[val_loader.dataset.indices[np.argmax(normal_scores)]]),
#        title_suffix=f"{best_run_result['evaluation_string']}\nValidation Test Set");
#    for name, ds in perturbed_dataset_loaders:
#        P.plot_perturbed_image(data.get_image_by_id(ds.dataset.keys[np.argmin(best_run_result['Ys_perturbed'][name])]),
#                               title_suffix=f"{best_run_result['evaluation_string']}\n{name} (score: {np.min(best_run_result['Ys_perturbed'][name]):0.3f})");
#
#    for name, ds in out_of_distribution_loaders:
#        P.plot_perturbed_image(data.get_image_by_id(ds.dataset.keys[np.argmin(best_run_result['Ys_ood'][name])]),
#                               title_suffix=f"{best_run_result['evaluation_string']}\n{name} (score: {np.min(best_run_result['Ys_ood'][name]):0.3f})");
