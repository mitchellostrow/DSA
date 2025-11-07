import numpy as np
from tqdm import tqdm
from .dmd import DMD
from .dmdc import DMDc
from .subspace_dmdc import SubspaceDMDc
from .stats import (
    measure_nonnormality_transpose,
    compute_all_stats,
    measure_transient_growth,
)
from .resdmd import compute_residuals
import matplotlib.pyplot as plt
from typing import Literal
import warnings

def split_train_test(data, train_frac=0.8):
    if isinstance(data, list):
        train_data = [d for i, d in enumerate(data) if i < int(train_frac * len(data))]
        test_data = [d for i, d in enumerate(data) if i >= int(train_frac * len(data))]
        dim = data[0].shape[-1]
    elif data.ndim == 3 and data.shape[0] == 1:
        train_data = data[:, int(train_frac * data.shape[1]) :]
        test_data = data[:, : int(train_frac * data.shape[1])]
        dim = data.shape[-1]
    else:
        train_data = data[: int(train_frac * data.shape[0])]
        test_data = (
            data[int(train_frac * data.shape[0]) :] if train_frac < 1.0 else train_data
        )
        dim = data.shape[-1]
    return train_data, test_data, dim


def sweep_ranks_delays(
    data,
    n_delays,
    ranks,
    control_data=None,
    train_frac=0.8,
    reseed=5,
    return_residuals=True,
    return_transient_growth=False,
    return_mse=False,
    error_space="X",
    model_class=['DMD', 'DMDc', 'SubspaceDMDc'][0],
    **model_kwargs,
):
    """
    Sweep over combinations of DMD ranks and delays, returning AIC, MASE, non-normality, and residuals.

    Parameters
    ----------
    data : np.ndarray
        Input data (trials, time, features).
    n_delays : iterable
        List or array of delays to sweep.
    ranks : iterable
        List or array of ranks to sweep.
    train_frac : float
        Fraction of data to use for training. If greater than or equal to 1, tests on the training set
    reseed : int
        Reseed for DMD prediction.
    return_residuals : bool
        Whether to return residuals.
    measure_transient_growth : bool
        Whether to measure transient growth (numerical abscissa and l2 norm).
    return_mse: bool
        Whether to return the mean squared error of the prediction in place of MASE
    dmd_kwargs : dict
        Additional keyword arguments for DMD.

    Returns
    -------
    all_aics, all_mases, all_nnormals, all_residuals, all_num_abscissa, all_l2norm : np.ndarray
        Arrays of results for each (delay, rank) pair.
    """
    if model_class in ['DMDc', 'SubspaceDMDc']:
        assert control_data is not None, "Control data is required for DMDc and SubspaceDMDc"

    train_data, test_data, dim = split_train_test(data, train_frac)
    train_control_data, test_control_data, dim_control = split_train_test(control_data, train_frac)

    all_aics, all_mases, all_nnormals, all_residuals, all_l2norm = [], [], [], [], []
    for nd in tqdm(n_delays):
        rresiduals = []
        aics, mases, nnormals, l2norms = [], [], [], []
        for r in ranks:
            if r is None or r > nd * dim:
                aics.append(np.inf)
                mases.append(np.inf)
                nnormals.append(np.inf)
                rresiduals.append(np.inf)
                l2norms.append(np.inf)
                continue

            if model_class == 'DMD':
                model = DMD(train_data, n_delays=nd, rank=r, **model_kwargs)
            elif model_class == 'DMDc':
                model = DMDc(train_data, train_control_data, n_delays=nd, rank_output=r, **model_kwargs)
            elif model_class == 'SubspaceDMDc':
                model = SubspaceDMDc(train_data, train_control_data, n_delays=nd, rank=r, **model_kwargs)
            else:
                raise ValueError(f"Invalid model class: {model_class}. Valid options are 'DMD', 'DMDc', and 'SubspaceDMDc'.")
            model.fit()
            
            # pred, H_test_pred, H_test_true, V_test_pred, V_test_true = dmd.predict(
            #     test_data, reseed=reseed, full_return=True
            # )
            if model_class == "DMD":
                pred, H_test_pred, H_test_true= model.predict(
                    test_data, reseed=reseed, full_return=True
                )
            elif model_class == "DMDc":
                pred, H_test_pred, H_test_true= model.predict(
                    test_data, test_control_data, reseed=reseed, full_return=True
                )
            else:
                pred = model.predict(test_data, test_control_data, reseed=reseed)

            if error_space == "H":
                if model_class == 'SubspaceDMDc':
                    raise ValueError("H space not implemented for SubspaceDMDc. Use X space instead.")
                pred = H_test_pred
                test_data_err = H_test_true
            elif error_space == "V":
                raise ValueError("V space not implemented ")
                # pred = V_test_pred
                # test_data_err = V_test_true
            elif error_space == "X":
                pred = pred
                test_data_err = test_data
            else:
                raise ValueError(f"Invalid error space: {error_space}")

            if hasattr(pred, "cpu"):
                pred = pred.cpu()
            if hasattr(test_data_err, "cpu"):
                test_data_err = test_data_err.cpu()

            if isinstance(pred, list):
                pred = np.concatenate(pred, axis=0)
                test_data_err = np.concatenate(test_data_err, axis=0)
            # if featurize and ndim is not None:
            # pred = pred[:, :, -ndim:]
            # stats = compute_all_stats(pred, test_data_err[:, :, -ndim:], dmd.rank)
            # else:
            stats = compute_all_stats(test_data_err, pred, model.rank if model_class in ['DMD', 'SubspaceDMDc'] else model.rank_output)
            aic = stats["AIC"]
            mase = stats["MASE"]
            if return_mse:
                mase = stats["MSE"]
            nnormal = measure_nonnormality_transpose(
                model.A_v.cpu().detach().numpy() if hasattr(model.A_v, "cpu") else model.A_v
            )
            if return_transient_growth:
                l2norm = measure_transient_growth(
                    model.A_v.cpu().detach().numpy()
                    if hasattr(model.A_v, "cpu")
                    else model.A_v
                )
            else:
                l2norm = None
            if return_residuals and model_class == 'DMD':
                L, G, residuals, _ = compute_residuals(model)
                residuals = np.mean(residuals)
            else:
                warnings.warn(f"Residuals not implemented for {model_class}")
                residuals = None
            aics.append(aic)
            mases.append(mase)
            nnormals.append(nnormal)
            rresiduals.append(residuals)
            l2norms.append(l2norm)
        all_aics.append(aics)
        all_mases.append(mases)
        all_nnormals.append(nnormals)
        all_residuals.append(rresiduals)
        all_l2norm.append(l2norms)
    all_aics = np.array(all_aics)
    all_mases = np.array(all_mases)
    all_nnormals = np.array(all_nnormals)
    all_residuals = np.array(all_residuals)
    all_l2norm = np.array(all_l2norm)

    return_tuples = [all_aics, all_mases, all_nnormals]
    if return_residuals:
        return_tuples.append(all_residuals)
    if return_transient_growth:
        return_tuples.append(all_l2norm)
    return tuple(return_tuples)


def plot_sweep_results(
    aics,
    mases,
    nnormals=None,
    residuals=None,
    l2norm=None,
    n_delays=None,
    ranks=None,
    name=None,
    save_path=None,
    figsize=(10, 4),
    return_mse=False,
    cmap="gist_gray",
    error_space="X",
):
    to_plot = [aics, mases]
    if nnormals is not None:
        to_plot.append(nnormals)
    if residuals is not None:
        to_plot.append(residuals)
    if l2norm is not None:
        to_plot.append(l2norm)
    fig, ax = plt.subplots(1, len(to_plot), figsize=figsize)
    cmap = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    scale_denom = len(aics) + 3
    for j in range(len(aics)):
        ax[0].plot(ranks, aics[j], label=f"{n_delays[j]}", color=cmap(j / scale_denom))
        ax[1].plot(ranks, mases[j], color=cmap(j / scale_denom), label=f"{n_delays[j]}")
        ax[1].axhline(1, color="black", linestyle="--")
        if nnormals is not None:
            ax[2].plot(
                ranks, nnormals[j], color=cmap(j / scale_denom), label=f"{n_delays[j]}"
            )
        if residuals is not None:
            ax[3].plot(
                ranks, residuals[j], color=cmap(j / scale_denom), label=f"{n_delays[j]}"
            )
        if l2norm is not None:
            ax[4].plot(
                ranks, l2norm[j], color=cmap(j / scale_denom), label=f"{n_delays[j]}"
            )
            ax[4].axhline(1, color="black", linestyle="--")

        ax[1].set_yscale("log")

        ax[0].set_ylabel(f"{error_space} AIC")
        ax[1].set_ylabel(
            f"{error_space} MASE" if not return_mse else f"{error_space} MSE"
        )
        if nnormals is not None:
            ax[2].set_ylabel(f"Non-normal score")
        if residuals is not None:
            ax[3].set_ylabel(f"Average residual of eigenvalues")
        if l2norm is not None:
            ax[4].set_ylabel(f"L2 norm of matrix")
        ax[-1].legend(
            title="# delays", loc="upper right", bbox_to_anchor=(2, 1), borderaxespad=1
        )
        for k in range(len(to_plot)):
            ax[k].set_xlabel("Rank")
            ax[k].spines["top"].set_visible(False)
            ax[k].spines["right"].set_visible(False)
    plt.suptitle(f"{name if name else ''}_tuning_{error_space}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}.pdf")
        # plt.close()
    else:
        return fig, ax


def predict_and_stats(dmd, test_data, reseed):
    pred, H_test_pred, H_test_true, V_test_pred, V_test_true = dmd.predict(
        test_data, reseed=reseed, full_return=True
    )
    if hasattr(pred, "cpu"):
        pred = pred.cpu()
    if hasattr(H_test_pred, "cpu"):
        H_test_pred = H_test_pred.cpu()
    if hasattr(H_test_true, "cpu"):
        H_test_true = H_test_true.cpu()
    if hasattr(V_test_pred, "cpu"):
        V_test_pred = V_test_pred.cpu()
    if hasattr(V_test_true, "cpu"):
        V_test_true = V_test_true.cpu()
    if isinstance(pred, list):
        pred = np.concatenate(pred, axis=0)
        test_data = np.concatenate(test_data, axis=0)
    xstats = compute_all_stats(test_data, pred, dmd.rank)
    hstats = compute_all_stats(H_test_true, H_test_pred, dmd.rank)
    vstats = compute_all_stats(V_test_true, V_test_pred, dmd.rank)
    return xstats, hstats, vstats


def sweep_ranks_delays_all_error_types(
    data,
    n_delays,
    ranks,
    train_frac=0.8,
    reseeds=5,
    return_type: Literal["tuple", "dict"] = "dict",
    **dmd_kwargs,
):
    """
    Sweep over combinations of DMD ranks and delays, returning all error types (AIC, MASE, MSE in X space, H space, V space, with and without reseeds)
    Will also compute non-normality of the DMD matrix.

    Parameters
    ----------
    data : np.ndarray
        Input data (trials, time, features).
    n_delays : iterable
        List or array of delays to sweep.
    ranks : iterable
        List or array of ranks to sweep.
    train_frac : float
        Fraction of data to use for training. If greater than or equal to 1, tests on the training set
    reseed : (int, list)
        Reseeds for DMD prediction.
    dmd_kwargs : dict
        Additional keyword arguments for DMD.

    Returns
    -------

        Arrays of results for each (delay, rank) pair.
    """
    train_data, test_data, dim = split_train_test(data, train_frac)

    if not isinstance(reseeds, list) and reseeds in set([1, "none", None, "", 0]):
        reseeds = [1]
    elif isinstance(reseeds, int):
        reseeds = [1, reseeds]
    if 1 not in reseeds:
        reseeds = [1] + reseeds

    def init_arr(d=3):
        if d == 3:
            arr = np.zeros((len(reseeds), len(n_delays), len(ranks)))
        elif d == 2:
            arr = np.zeros((len(n_delays), len(ranks)))
        arr[:] = np.nan
        return arr

    all_aicsx_reseed, all_masesx_reseed, all_msesx_reseed = (
        init_arr(),
        init_arr(),
        init_arr(),
    )
    all_aicsh_reseed, all_masesh_reseed, all_msesh_reseed = (
        init_arr(),
        init_arr(),
        init_arr(),
    )
    all_aicsv_reseed, all_masesv_reseed, all_msesv_reseed = (
        init_arr(),
        init_arr(),
        init_arr(),
    )

    for i, nd in tqdm(enumerate(n_delays)):
        for j, r in enumerate(ranks):
            if r is None or r > nd * dim:
                continue
            dmd = DMD(train_data, n_delays=nd, rank=r, **dmd_kwargs)
            dmd.fit()
            for k, reseed in enumerate(reseeds):
                xstats, hstats, vstats = predict_and_stats(dmd, test_data, reseed)
                all_aicsx_reseed[k, i, j] = xstats["AIC"]
                all_masesx_reseed[k, i, j] = xstats["MASE"]
                all_msesx_reseed[k, i, j] = xstats["MSE"]

                all_aicsh_reseed[k, i, j] = hstats["AIC"]
                all_masesh_reseed[k, i, j] = hstats["MASE"]
                all_msesh_reseed[k, i, j] = hstats["MSE"]

                all_aicsv_reseed[k, i, j] = vstats["AIC"]
                all_masesv_reseed[k, i, j] = vstats["MASE"]
                all_msesv_reseed[k, i, j] = vstats["MSE"]

    if return_type == "tuple":
        return (
            all_aicsx_reseed,
            all_masesx_reseed,
            all_msesx_reseed,
            all_aicsh_reseed,
            all_masesh_reseed,
            all_msesh_reseed,
            all_aicsv_reseed,
            all_masesv_reseed,
            all_msesv_reseed,
        )
    elif return_type == "dict":
        return {
            "reseeds": reseeds,
            "aicsx_reseed": all_aicsx_reseed,
            "masesx_reseed": all_masesx_reseed,
            "msesx_reseed": all_msesx_reseed,
            "aicsh_reseed": all_aicsh_reseed,
            "masesh_reseed": all_masesh_reseed,
            "msesh_reseed": all_msesh_reseed,
            "aicsv_reseed": all_aicsv_reseed,
            "masesv_reseed": all_masesv_reseed,
            "msesv_reseed": all_msesv_reseed,
            "n_delays": n_delays,
            "ranks": ranks,
        }


def plot_sweep_results_all_error_types(
    return_dict,
    name=None,
    save_path=None,
    figsize=(2, 4),
    xscale="log",
    aic_scale="symlog",
    mase_scale="log",
    plot_herror=False,
    new_plot_reseeds=False,
    cmap="gist_gray",
    metrics_order=["AIC", "MASE", "MSE"],
    pretty_yticks=False,
):
    """
    Plot all error types from sweep_ranks_delays_all_error_types as a 3 x (3*len(reseeds)) grid,
    or, if separate_by_space is True, make 3 separate plots (one for each of X, H, V), with columns as metrics and rows as reseeds.

    Parameters
    ----------
    return_dict : dict
        Output from sweep_ranks_delays_all_error_types.
    name : str or None
        Title for the plot.
    save_path : str or None
        If provided, save the figure to this path (as .pdf).
    figsize : tuple
        Figure size.
    column_order : {'by_reseed', 'by_metric'}
        If 'by_reseed', columns are [aics[0], mases[0], mses[0], aics[1], mases[1], mses[1], ...] (grouped by reseed).
        If 'by_metric', columns are [aics[0], aics[1], ..., mases[0], mases[1], ..., mses[0], mses[1], ...] (grouped by metric).
    plot_herror : bool
    new_plot_reseeds : bool
        If True, plot the reseeds in a new plot, with the same number of columns
    """
    fig_axes = []
    if new_plot_reseeds:
        return_dict_new = {}
        return_dict_plot = {}
        for k, v in return_dict.items():
            if (isinstance(v, np.ndarray) and v.size == 0) or (
                isinstance(v, list) and len(v) == 0
            ):
                return_dict_new[k] = []
                return []
            elif k in ["n_delays", "ranks"]:
                return_dict_new[k] = v
                return_dict_plot[k] = v
            else:
                return_dict_new[k] = v[1:]
                return_dict_plot[k] = v[0:1]
        fig_axes = plot_sweep_results_all_error_types(
            return_dict_new,
            name=name,
            save_path=save_path,
            figsize=figsize,
            xscale=xscale,
            aic_scale=aic_scale,
            plot_herror=plot_herror,
            new_plot_reseeds=new_plot_reseeds,
            metrics_order=metrics_order,
        )
        return_dict = return_dict_plot
    reseeds = return_dict["reseeds"]
    n_delays = return_dict["n_delays"]
    ranks = return_dict["ranks"]
    all_aicsx_reseed = return_dict["aicsx_reseed"]
    all_masesx_reseed = return_dict["masesx_reseed"]
    all_msesx_reseed = return_dict["msesx_reseed"]
    all_aicsh_reseed = return_dict["aicsh_reseed"]
    all_masesh_reseed = return_dict["masesh_reseed"]
    all_msesh_reseed = return_dict["msesh_reseed"]
    all_aicsv_reseed = return_dict["aicsv_reseed"]
    all_masesv_reseed = return_dict["masesv_reseed"]
    all_msesv_reseed = return_dict["msesv_reseed"]

    n_reseeds = len(reseeds)
    metrics = ["AIC", "MASE", "MSE"]
    spaces = ["X", "V"] if not plot_herror else ["X", "H", "V"]
    data_arrays = [
        [all_aicsx_reseed, all_aicsv_reseed]
        + ([all_aicsh_reseed] if plot_herror else []),
        [all_masesx_reseed, all_masesv_reseed]
        + ([all_masesh_reseed] if plot_herror else []),
        [all_msesx_reseed, all_msesv_reseed]
        + ([all_msesh_reseed] if plot_herror else []),
    ]
    data_arrays = [data_arrays[metrics.index(metric)] for metric in metrics_order]
    metrics = metrics_order

    cmap = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    figs_axes = []
    for space_idx, space in enumerate(spaces):
        # Each plot: rows = metrics, cols = reseeds (transpose from original)
        fig, axes = plt.subplots(
            len(metrics),
            n_reseeds,
            figsize=(figsize[0] * n_reseeds, figsize[1]),
            sharex=True,
            sharey="row",
        )
        if len(reseeds) == 1:
            if len(metrics) == 1:
                axes = [axes]
            else:
                axes = axes.reshape(-1, 1)
        if len(metrics) == 1:
            axes = [axes]

        # For storing y-limits for each metric row
        row_ymins = [np.inf] * len(metrics)
        row_ymaxs = [-np.inf] * len(metrics)
        for metric_idx, metric in enumerate(metrics):
            for reseed_idx, reseed in enumerate(reseeds):
                arr = data_arrays[metric_idx][space_idx]
                ax = axes[metric_idx][reseed_idx]
                for nd_idx, nd in enumerate(n_delays):
                    y = arr[reseed_idx, nd_idx]
                    ax.plot(
                        ranks,
                        y,
                        label=f"{nd}",
                        color=cmap(nd_idx / (len(n_delays) + 3)),
                    )
                    # Update min/max for this row, ignoring nan
                    valid_y = np.asarray(y)
                    valid_y = valid_y[np.isfinite(valid_y)]
                    if valid_y.size > 0:
                        row_ymins[metric_idx] = min(
                            row_ymins[metric_idx], np.nanmin(valid_y)
                        )
                        row_ymaxs[metric_idx] = max(
                            row_ymaxs[metric_idx], np.nanmax(valid_y)
                        )
                if metric == "MASE":
                    ax.axhline(1, color="black", linestyle="--", linewidth=0.7)
                if metric in {"MASE", "MSE"} and mase_scale in {
                    "symlog",
                    "log",
                    "linear",
                }:
                    ax.set_yscale(mase_scale)
                if aic_scale in {"symlog", "log", "linear"} and metric == "AIC":
                    ax.set_yscale(aic_scale)
                if xscale == "log":
                    ax.set_xscale("log")
                if reseed_idx == 0:
                    ax.set_ylabel(f"{space} {metric}", fontsize=10)
                else:
                    ax.set_ylabel("")
                if metric_idx == 0:
                    ax.set_title(f"reseed {reseed}")
                else:
                    ax.set_title("")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                axes[-1][reseed_idx].set_xlabel("Rank")
                if metric_idx == 0 and reseed_idx == len(reseeds) - 1:
                    ax.legend(
                        title="# delays",
                        fontsize=12,
                        loc="upper right",
                        bbox_to_anchor=(2.3, 1.2),
                        borderaxespad=1,
                    )

        # Set yticks for each row to be the min and max (rounded) of all the plots on that row
        for metric_idx in range(len(metrics)):
            ymin = (
                0.75 * row_ymins[metric_idx]
                if row_ymins[metric_idx] > 0
                else 1.25 * row_ymins[metric_idx]
            )
            ymax = (
                1.25 * row_ymaxs[metric_idx]
                if row_ymaxs[metric_idx] > 0
                else 0.75 * row_ymaxs[metric_idx]
            )
            for reseed_idx in range(n_reseeds):
                ax = axes[metric_idx][reseed_idx]
                # Remove all yticks and labels before setting new ones
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.yaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_minor_locator(plt.NullLocator())
            if pretty_yticks:
                for reseed_idx in range(n_reseeds):
                    ax = axes[metric_idx][reseed_idx]
                    # Only set exactly two yticks (min and max), and always set their labels
                    ax.set_ylim([ymin, ymax])
                    ax.set_yticks([ymin, ymax])
                    # Set tick labels to formatted numbers (scientific if needed)
                    ticklabels = [f"{ymin:.2g}", f"{ymax:.2g}"]
                    ax.set_yticklabels(ticklabels)

        plt.suptitle(f"{name + '_' if name else ''}{space} tuning", fontsize=14, y=1.05)
        plt.tight_layout()  # rect=[0, 0, 1, 0.97])
        if save_path is not None:
            plt.savefig(
                f"{save_path}_{space}_metrics_{metrics_order}_reseeds{reseeds}.pdf"
            )
            # plt.close()
        figs_axes.append((fig, axes))

    return figs_axes + fig_axes
