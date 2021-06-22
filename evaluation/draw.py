import numpy as np
import torch
from itertools import cycle
import random
from numpy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from .confusion_matrix_pretty_print import pretty_plot_confusion_matrix

COLOR_PALETTES = ["#0173B2", "#D55E00", "#029E73", "#CC78BC", "#ECE133", "#56B4E9", "#DE8F05"]
# COLOR_PALETTES = cycle(sns.palettes.SEABORN_PALETTES['colorblind6'])
def draw_TSNE(features, y_true, label_names, save_path, title=None, tsne_plot_count=1200):
    # t-SNE and PCA plots#
    # np.random.seed(2021)
    # random.seed(123)
    assert features.shape[0] == len(y_true)
    plt.figure(figsize=(8, 8))
    classes = np.unique(y_true)
    if tsne_plot_count > len(y_true):
        tsne_plot_count = len(y_true)
    # sample n = tsne_plot_count features
    sample_list = [i for i in range(len(y_true))]
    sample_list = random.sample(sample_list, tsne_plot_count)
    feature = [features[i].cpu().detach().numpy() for i in sample_list]
    y_labels = [y_true[i] for i in sample_list]
    # Dimensionality Reduction
    tsne_result = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=800, learning_rate=300).fit_transform(feature)
    data = {'x': np.array(tsne_result[:, 0]), 'y': np.array(tsne_result[:, 1]), 'label': np.array(y_labels)}
    for c in classes:
        plt.scatter(data['x'][data['label'] == c], data['y'][data['label'] == c],
                    c=COLOR_PALETTES[int(c)], marker='o', s=40)
    plt.legend(labels=label_names, loc="lower right")
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(save_path + '/t-SNE.pdf')
    # sns.scatterplot(
    #     x="x", y="y",
    #     hue="label",
    #     palette=sns.color_palette("dark6", 5),
    #     data=data,
    #     legend="brief"
    # )
    # plt.legend(loc="upper right")
    # plt.show()

def draw_ROC(logits, y_true, label_names, save_path, title=None):
    classes = np.unique(y_true)
    n_classes = len(np.unique(y_true))
    y_score = torch.nn.Softmax(dim=1)(logits.cpu()) if n_classes > 2 else torch.nn.Softmax(dim=1)(logits.cpu())[:, 1]
    y_true = label_binarize(y_true, classes=classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i]) if n_classes > 2 else roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    if n_classes != 2:
        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.detach().numpy().ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)
        #
        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (AUC = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          color='navy', linestyle=':', linewidth=4)

        lw = 2
        colors = cycle(sns.palettes.SEABORN_PALETTES['colorblind6'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                     ''.format(label_names[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    else:
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path + '/ROC.pdf')


N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1 / 256, 1, N)
vals[:, 1] = np.linspace(115 / 256, 1, N)
vals[:, 2] = np.linspace(178 / 256, 1, N)
newcmp = ListedColormap(ListedColormap(vals).colors[::-1])
def draw_cm(y_true, y_predict, label_names, save_path, pretty_print=True, title=None):
    print(y_true.shape)
    print(y_predict.shape)
    cm = confusion_matrix(y_true, y_predict)
    if not pretty_print:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label_names)
        disp.plot(cmap=newcmp, colorbar=False)
        if title:
            disp.ax_.set_title(title)
        # plt.savefig('cm_plot.pdf')
    else:
        import pandas as pd
        cm = pd.DataFrame(cm, columns=label_names)
        pretty_plot_confusion_matrix(cm)

    plt.savefig(save_path + '/confusion_matrix.pdf')