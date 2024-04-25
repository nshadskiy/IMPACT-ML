import matplotlib.pyplot as plt
import logging
import numpy as np
import mplhep as hep
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from NeuralNets import Network

log = logging.getLogger("training")
plt.style.use(hep.style.CMS)

#########################################################################################
### plot functions ###
#########################################################################################
# KIT colors: gruen=#009682 , blau=#4664aa , maigruen=#8cb63c , gelb=#fce500 , orange=#df9b1b , braun=#a7822e , rot=#a22223 , lila=#a3107c , cyanblau=#19a1e0


def loss(network: Network) -> None:
    plt.plot(
        range(1, len(network.train_loss_log) + 1, 1),
        network.train_loss_log,
        color="blue",
        label="train",
    )
    plt.plot(
        range(1, len(network.val_loss_log) + 1, 1),
        network.val_loss_log,
        color="orange",
        label="validation",
    )
    plt.axvline(x=network.best_epoch, color="red", ls="--", label="best epoch")

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Loss", fontsize=24)

    plt.grid()
    plt.legend()

    plt.savefig(network.save_path + "/loss.pdf")
    plt.savefig(network.save_path + "/loss.png")
    plt.close()

    log.info("-" * 50)
    log.info("save loss function in " + network.save_path)


# def hist_errorbar(data, hist, bin_edges, weights=None, bins=10, plt_range=[0,1], scale=None, label="label"):
#     if not scale==None:
#         hist, bin_edges = np.histogram(data, weights=weights*scale, bins=bins, range=plt_range)

#     if label=="ttH":
#         plt.hist( data, weights=weights*scale, bins=bins, range=plt_range, histtype='step', color='#009682', label=label+" x {}".format(round(scale)), zorder=1 )
#         plt.errorbar(0.5*(bin_edges[1:]+bin_edges[:-1]), hist, yerr=np.sqrt(hist), color='#009682', fmt='none')
#     else:
#         plt.hist( data, weights=weights, bins=bins, range=plt_range, histtype='step', color='black', fill=True, facecolor='#a22223', label=label, zorder=0 )
#         plt.errorbar(0.5*(bin_edges[1:]+bin_edges[:-1]), hist, yerr=np.sqrt(hist), color='black', fmt='none')


# def plot_prediction(network):
#     signal     = torch.max( network.prediction.data, 1 )[0].cpu().numpy()[ network.y_test.cpu().numpy()==1 ]
#     background = torch.max( network.prediction.data, 1 )[0].cpu().numpy()[ network.y_test.cpu().numpy()==0 ]
#     sig_weights = network.pred_weights[network.y_test.cpu().numpy()==1]
#     bkg_weights = network.pred_weights[network.y_test.cpu().numpy()==0]

#     plt.figure()

#     ax = plt.axes()
#     plt.xlim(0,1)
#     plt.xticks(np.arange(0,1.01,0.1))
#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.yaxis.set_minor_locator(AutoMinorLocator())
#     ax.tick_params(direction='in', which='both', top=True, right=True, zorder=2)

#     ax.text(0.99, 1.01, "ROC-AUC={:.3f}".format(network.roc_auc_score), fontsize=11, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

#     hist_bins = 40

#     hist_bkg, edge_bkg = np.histogram(background, weights=bkg_weights, bins=hist_bins, range=[0,1])
#     hist_sig, edge_sig = np.histogram(signal    , weights=sig_weights, bins=hist_bins, range=[0,1])
#     sig_scale = sum(hist_bkg)/sum(hist_sig)

#     hist_errorbar(background, hist_bkg, edge_bkg, weights=bkg_weights, bins=hist_bins, label="background")
#     hist_errorbar(signal    , hist_sig, edge_sig, weights=sig_weights, bins=hist_bins, scale=sig_scale, label="ttH")

#     plt.ylim(ymin=0., ymax=max( max(hist_bkg),max(hist_sig) )*1.2)

#     plt.xlabel("Discriminant", fontsize=16, loc='right')
#     plt.ylabel("Events", fontsize=16, loc='top')

#     plt.legend(frameon=False, fontsize=14)

#     plt.savefig( network.save_path + "/prediction.pdf" )
#     plt.savefig( network.save_path + "/prediction.png" )
#     plt.close()

#     log.info( "-"*50 )
#     log.info( "save discriminant at " + network.save_path )


def multiclass_nodes(network: Network) -> None:
    for comb in network.data.plot_mass_combinations:
        for cl in network.data.classes:
            node = network.data.label_dict[cl]
            node_pred = (
                network.prediction[f"massX_{comb['massX']}_massY_{comb['massY']}"]
                .cpu()
                .numpy()[:, node]
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            for proc in network.data.classes:
                pred = node_pred[
                    network.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
                    .cpu()
                    .numpy()
                    == network.data.label_dict[proc]
                ]
                pred_weights = network.pred_weights[
                    f"massX_{comb['massX']}_massY_{comb['massY']}"
                ][
                    network.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
                    .cpu()
                    .numpy()
                    == network.data.label_dict[proc]
                ]
                h, edges = np.histogram(
                    pred, bins=15, range=(0.0, 1.0), weights=pred_weights
                )
                hep.histplot(h, bins=edges, ax=ax, label=proc, density=True, yerr=False)

            ax.legend()
            plt.xlabel(f"{cl} node", fontsize=24)
            plt.ylabel("normalized", fontsize=24)
            ax.text(
                0.99,
                1.01,
                "ROC-AUC={:.3f}".format(
                    network.roc_auc_scores[
                        f"massX_{comb['massX']}_massY_{comb['massY']}"
                    ][network.data.label_dict[cl]]
                ),
                fontsize=20,
                fontweight="bold",
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )

            plt.savefig(
                network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_node_{cl}.pdf"
            )
            plt.savefig(
                network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_node_{cl}.png"
            )
            plt.close()

            log.info("-" * 50)
            log.info(
                "save node output to "
                + network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_node_{cl}.pdf"
            )


def multiclass_classes(network: Network) -> None:
    for comb in network.data.plot_mass_combinations:
        class_idx = np.argmax(
            network.prediction[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            .cpu()
            .numpy(),
            axis=1,
        )
        class_pred = np.amax(
            network.prediction[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            .cpu()
            .numpy(),
            axis=1,
        )

        for cl in network.data.classes:
            fig, ax = plt.subplots(figsize=(10, 8))
            bkg_hist_list = list()
            bkg_hist_label = list()
            for proc in network.data.classes:
                if proc not in network.config["signal"]:
                    pred = class_pred[
                        (class_idx == network.data.label_dict[cl])
                        & (
                            network.y_test[
                                f"massX_{comb['massX']}_massY_{comb['massY']}"
                            ]
                            .cpu()
                            .numpy()
                            == network.data.label_dict[proc]
                        )
                    ]
                    pred_weights = network.pred_weights[
                        f"massX_{comb['massX']}_massY_{comb['massY']}"
                    ][
                        (class_idx == network.data.label_dict[cl])
                        & (
                            network.y_test[
                                f"massX_{comb['massX']}_massY_{comb['massY']}"
                            ]
                            .cpu()
                            .numpy()
                            == network.data.label_dict[proc]
                        )
                    ]
                    h, edges = np.histogram(
                        pred, bins=15, range=(0.0, 1.0), weights=pred_weights
                    )
                    bkg_hist_list.append(h)
                    bkg_hist_label.append(proc)
            hep.histplot(
                bkg_hist_list,
                bins=edges,
                stack=True,
                histtype="fill",
                ax=ax,
                label=bkg_hist_label,
                density=False,
                yerr=False,
            )

            for proc in network.data.classes:
                if proc in network.config["signal"]:
                    pred = class_pred[
                        (class_idx == network.data.label_dict[cl])
                        & (
                            network.y_test[
                                f"massX_{comb['massX']}_massY_{comb['massY']}"
                            ]
                            .cpu()
                            .numpy()
                            == network.data.label_dict[proc]
                        )
                    ]
                    pred_weights = network.pred_weights[
                        f"massX_{comb['massX']}_massY_{comb['massY']}"
                    ][
                        (class_idx == network.data.label_dict[cl])
                        & (
                            network.y_test[
                                f"massX_{comb['massX']}_massY_{comb['massY']}"
                            ]
                            .cpu()
                            .numpy()
                            == network.data.label_dict[proc]
                        )
                    ]
                    h, edges = np.histogram(
                        pred, bins=15, range=(0.0, 1.0), weights=pred_weights
                    )
                    hep.histplot(
                        h, bins=edges, ax=ax, label=proc, density=False, yerr=False
                    )

            ax.legend()
            plt.xlabel(f"{cl} class", fontsize=24)

            plt.savefig(
                network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_class_{cl}.pdf"
            )
            plt.savefig(
                network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_class_{cl}.png"
            )
            plt.close()

            log.info("-" * 50)
            log.info(
                "save discriminant to "
                + network.save_path
                + f"/massX_{comb['massX']}_massY_{comb['massY']}"
                + f"/prediction_class_{cl}.pdf"
            )


def confusion(network: Network) -> None:
    for comb in network.data.plot_mass_combinations:
        class_idx = np.argmax(
            network.prediction[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            .cpu()
            .numpy(),
            axis=1,
        )

        cm_eff = confusion_matrix(
            y_true=network.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            .cpu()
            .numpy(),
            y_pred=class_idx,
            labels=np.array(list(network.data.label_dict.values())),
            normalize="true",
        )
        cm_eff = np.round(cm_eff, 2)
        log.info(f"Confusion matrix (efficiency): {cm_eff}")

        cm_eff_plot = ConfusionMatrixDisplay(
            confusion_matrix=cm_eff, display_labels=network.data.classes
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_eff_plot.plot(ax=ax, xticks_rotation=65, colorbar=False)
        plt.tight_layout()

        plt.savefig(
            network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_eff.pdf"
        )
        plt.savefig(
            network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_eff.png"
        )
        plt.close()

        log.info("-" * 50)
        log.info(
            "save confusion to "
            + network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_eff.pdf"
        )
        
        cm_pur = confusion_matrix(
            y_true=network.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            .cpu()
            .numpy(),
            y_pred=class_idx,
            labels=np.array(list(network.data.label_dict.values())),
            normalize="pred",
        )
        cm_pur = np.round(cm_pur, 2)
        log.info(f"Confusion matrix (purity): {cm_pur}")

        cm_pur_plot = ConfusionMatrixDisplay(
            confusion_matrix=cm_pur, display_labels=network.data.classes
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_pur_plot.plot(ax=ax, xticks_rotation=65, colorbar=False)
        plt.tight_layout()

        plt.savefig(
            network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_pur.pdf"
        )
        plt.savefig(
            network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_pur.png"
        )
        plt.close()

        log.info("-" * 50)
        log.info(
            "save confusion to "
            + network.save_path
            + f"/massX_{comb['massX']}_massY_{comb['massY']}"
            + "/confusion_matrix_pur.pdf"
        )
