import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
from tqdm import tqdm
import math
import warnings
import clip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
from matplotlib import rc
from data_utils import get_loaders, get_concepts, get_feature_dir
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_average_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correlations = []

    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        correlations.append(corr)

    return np.mean(correlations)


def concept_activation_per_class(loader, disc, num_concepts,
                                 num_classes, concepts, classes,
                                 base_dir='saved_models', prior=True):
    conc_act = torch.zeros([num_concepts, num_classes])
    disc.eval()
    with torch.no_grad():
        count_per_class = np.zeros([num_classes])
        for batch, (images, labels) in enumerate(loader):
            labels = labels.type(torch.LongTensor)  # casting to long

            images = images.to('cuda', non_blocking=True)
            labels = labels

            feats = images  # / images.norm(dim=-1, keepdim=True)

            # probs in R^{batch, concepts}
            probs, _ = disc(feats, probs_only=True)
            probs = probs.cpu().numpy()
            # set the very low probabilities to zero
            probs[np.where(probs < 1e-2)] = 0.

            if np.any(probs > 1.):
                raise ValueError('what, probs >1?')

            # add the contribution of each example to the respective entries

            for i in range(num_classes):
                batch_class_i = probs[np.where(labels == i)]
                count_per_class[i] += batch_class_i.shape[0]
                conc_act[:, i] += batch_class_i.sum(0)
        conc_act /= count_per_class

        results_dir = base_dir + 'prior_figs/' * prior + 'posterior_figs/' * (not prior)
        os.makedirs(results_dir, exist_ok=True)

        np.savetxt(results_dir + "concept_acts_per_class.csv", conc_act,
                   delimiter=",", fmt='%10.5f')
        # concept per fig
        cpf = 40
        for i in range(int(conc_act.shape[0] / cpf) + 1):
            cur_act = conc_act[i * cpf: (i + 1) * cpf]
            cur_conc = concepts[i * cpf: (i + 1) * cpf]
            lencur = cur_act.shape[0]

            fig = plt.figure(figsize=(15, 15))
            ax = sns.heatmap(cur_act.T, cmap='binary', linewidth=0.5,
                             square=True, vmin=0.0, vmax=1.0,
                             cbar_kws={"shrink": .95, "ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                                       'aspect': 50, 'pad': 0.01})

            plt.xticks(np.arange(lencur) + 0.5, cur_conc,
                       rotation=90, fontsize="8")
            plt.yticks(np.arange(num_classes) + 0.5, classes,
                       rotation=0, fontsize="8", va="center")

            # ax.set_yticklabels(classes, minor=True)
            plt.tight_layout()
            plt.savefig(results_dir + 'batch_{}.pdf'.format(i), bbox_inches="tight")
            plt.close(fig)


def concept_activation_per_example(loader, disc, classifier, num_concepts,
                                   concepts, classes, text_features, dataset,
                                   base_dir='saved_models', ):
    disc.eval()
    classifier.eval()

    # inds = np.random.choice(len(loader.dataset), 10)

    data = loader.dataset
    inds = np.random.choice(np.arange(13950, 13953, 1), 3, replace=False)
    results_dir = base_dir + 'post_analysis/'
    os.makedirs(results_dir, exist_ok=True)

    indices = 'imagenet_indices.csv' if 'imagenet' in dataset else 'cub_indices.csv'
    with open(indices, 'r') as f:
        csvreader = csv.reader(f)
        count = 0
        fpaths = {}
        for row in csvreader:
            if count == 0:
                count = 1
                continue
            fpaths[int(row[0])] = row[1]

    # for each example get the decision, get the weights corresponding to the class
    # and then get the concept contribution from that
    with torch.no_grad():
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for ind in inds:

            cur_feats, cur_label = data[ind]
            print(ind)
            print(cur_feats.shape)
            print(cur_label)
            cur_feats = cur_feats.to('cuda')
            cur_label = cur_label.to('cuda')
            mask, _ = disc(cur_feats, probs_only=True)
            cur_feats /= cur_feats.norm(dim=-1, keepdim=True)
            similarity = (cur_feats @ text_features.T)

            pred = torch.nn.functional.softmax(classifier(similarity, mask=mask), dim=-1)

            class_ind = torch.argmax(pred)
            class_name = classes[class_ind]
            class_name = ''.join([i for i in class_name if (not i.isdigit()) and (i != '.')])
            class_name = class_name.replace('_', ' ')
            true_class = classes[int(cur_label.cpu().numpy())]
            true_class = ''.join([i for i in true_class if not i.isdigit() and (i != '.')])
            true_class = true_class.replace('_', ' ')

            if class_name != true_class:
                continue

            spec_dir = results_dir + 'ind_{}/'.format(ind)
            os.makedirs(spec_dir, exist_ok=True)

            # get the original image if cub or imagenet
            if dataset in ['imagenet', 'cub']:
                fpath = fpaths[ind]
                fname = fpath.split('/')[-1]

                shutil.copyfile(fpath, spec_dir + fname)

            mask[torch.where(mask < 0.01)] = 0.

            # get the relevant weights and use the mask to filter
            # the mask should be probs to reflect the contribution
            rel_weights = (classifier.W[:, class_ind] * mask).cpu().numpy()
            rel_bias = classifier.bias[class_ind].cpu().numpy()
            rel_conf = pred[class_ind].cpu().numpy()
            mask = mask.cpu().numpy().astype(np.float32)
            similarity = similarity.cpu().numpy()
            contribution = similarity * rel_weights
            sparsity = (mask > 0.01).sum() / num_concepts

            sort_inds = np.argsort(contribution)[::-1]
            print(len(sort_inds))
            print(num_concepts)
            print(len(mask))
            print(len(similarity))
            print(len(rel_weights))
            print(len(contribution))
            print(len(concepts))

            with open(spec_dir + 'stats.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])
                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                     rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', (mask > 0.).sum() / num_concepts])

            with open(spec_dir + 'stats_only_active.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])

                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    if mask[sort_inds[i]] > 0.01:
                        writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                         rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', sparsity])

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig, ax = plt.subplots(layout='constrained', figsize=(18, 6))

            # Example data
            concepts_top = (concepts[sort_inds[0]],
                            concepts[sort_inds[1]],
                            concepts[sort_inds[2]],
                            concepts[sort_inds[3]],
                            'NOT ' * (contribution[sort_inds[-4]] < 0) + concepts[sort_inds[-4]],
                            'NOT ' * (contribution[sort_inds[-3]] < 0) + concepts[sort_inds[-3]],
                            'NOT ' * (contribution[sort_inds[-2]] < 0) + concepts[sort_inds[-2]],
                            'NOT ' * (contribution[sort_inds[-1]] < 0) + concepts[sort_inds[-1]]
                            )
            y_pos = np.arange(len(concepts_top))
            contribution = np.abs(contribution)
            contrib = contribution[sort_inds[:4]].tolist() + contribution[sort_inds[-4:]].tolist()
            colors = ['darkkhaki'] * 4 + ['darksalmon'] * 4
            hbars = ax.barh(y_pos, contrib, align='center',
                            height=0.5, color=colors)
            ax.set_yticks(y_pos, labels=concepts_top)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Concept Contribution')
            ax.set_title('Pred: {} - Conf: {:.5f} - Sparsity: {:.2f}\%'.format(class_name, rel_conf, 100 * sparsity))
            ax.bar_label(hbars, fmt='%.4f')
            ax.margins(x=0.1)

            plt.tight_layout()
            plt.savefig(spec_dir + 'contribution.pdf')  # , bbox_inches="tight")
            plt.close(fig)