"""
draw figures in paper
"""
import os
import re
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def tsne_hidden_layers():
    """
    1. use tsne to reduce dimension of each hidden layers
    2. plot hidden layers of clean vs adv
    """
    tsne = TSNE(n_components=2, perplexity=8, random_state=444)
    pca = PCA(n_components=10, random_state=444)
    labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # [1,2,1,2] or [1,2,3,4,1,2,3,4]
    layers = ['low_level', 'layer2', 'layer3', 'out']
    files = os.listdir(f'results/hidden_layers')
    batchs = [int(re.match(r'\d+', filename).group()) for filename in files]
    for i in tqdm(range(0, max(batchs), 4), desc=f'batches'):
        clean_features_1 = torch.load(f'results/hidden_layers/{i}_clean.pth')
        adv_features_1 = torch.load(f'results/hidden_layers/{i}_adv.pth')
        clean_features_2 = torch.load(f'results/hidden_layers/{i+1}_clean.pth')
        adv_features_2 = torch.load(f'results/hidden_layers/{i+1}_adv.pth')
        clean_features_3 = torch.load(f'results/hidden_layers/{i+2}_clean.pth')
        adv_features_3 = torch.load(f'results/hidden_layers/{i+2}_adv.pth')
        clean_features_4 = torch.load(f'results/hidden_layers/{i+3}_clean.pth')
        adv_features_4 = torch.load(f'results/hidden_layers/{i+3}_adv.pth')
        fig, axes = plt.subplots(1,4,figsize=(14, 4))
        for j, l in enumerate(layers):
            clean_li = clean_features_1[l]
            adv_li = adv_features_1[l]
            clean_li_2 = clean_features_2[l]
            adv_li_2 = adv_features_2[l]
            clean_li_3 = clean_features_3[l]
            adv_li_3 = adv_features_3[l]
            clean_li_4 = clean_features_4[l]
            adv_li_4 = adv_features_4[l]
            cat_tensor = torch.cat(
                (clean_li,clean_li_2,clean_li_3,clean_li_4,adv_li,adv_li_2,
                 adv_li_3,adv_li_4), dim=0).detach().cpu().numpy()
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], -1)
            del clean_li, adv_li, clean_li_2, adv_li_2

            cat_tensor = pca.fit_transform(cat_tensor)
            x_tsne = tsne.fit_transform(cat_tensor)
            df = pd.DataFrame(x_tsne, columns=["comp1", "comp2"])
            df["y"] = labels
            # if j == 0:
            #     sns_plot = sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
            #         palette=sns.color_palette("hls", cat_tensor.shape[0]),
            #         data=df, ax=axes[j])
            # else:
            sns_plot = sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", cat_tensor.shape[0]//2),
                data=df, ax=axes[j], legend=False)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
            axes[j].set_xlabel('')
            axes[j].set_ylabel('')

        plt.savefig(f"results/visualize/t-SNE_{i}.png",bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)


def infer_robust_model():
    """
    infer generated masks on the robust model
    """


if __name__ == '__main__':
    tsne_hidden_layers()
