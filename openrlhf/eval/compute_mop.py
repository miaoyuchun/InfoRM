# # Copyright (c) OpenMMLab. All rights reserved.
# source /mnt/code/users/liamding/tools/conda_install/anaconda3/bin/activate openrlhf_vllm072_ds015_inform
# export NUMBA_NUM_THREADS=32

import argparse
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from scipy.stats import chi2
from umap import UMAP
from sklearn.covariance import LedoitWolf
import pandas as pd

def get_gaussian_mask(normal_vectors: np.ndarray, alpha: float = 0.01, reg_eps: float = 1e-3):
    """
    对 normal_vectors 进行马氏距离检验，返回一个布尔 mask，表示哪些样本近似服从高斯分布。

    Args:
        normal_vectors (np.ndarray): 原始样本表征数组，shape = (N, D)
        alpha (float): 显著性水平，默认 0.01
        reg_eps (float): 协方差正则化项，默认 1e-3
    
    Returns:
        mask (np.ndarray): 布尔型数组，shape = (N,)；True 表示保留，False 表示异常样本
    """
    N, D = normal_vectors.shape

    lw = LedoitWolf().fit(normal_vectors)
    mu = lw.location_
    cov_inv = np.linalg.inv(lw.covariance_ + reg_eps * np.eye(normal_vectors.shape[1]))

    # 计算马氏距离平方
    delta = normal_vectors - mu
    d_squared = np.sum((delta @ cov_inv) * delta, axis=1)

    # 转换为 p 值
    p_values = 1 - chi2.cdf(d_squared, df=D)

    # 生成 mask：True 表示服从高斯分布
    mask = p_values >= alpha

    return mask


def compute_hacking_pvalues(normal_vectors: np.ndarray, test_vectors: np.ndarray, reg_eps: float = 1e-3):
    """
    Compute the p-value indicating whether a test sample is “hacking” by combining the Mahalanobis distance with the chi-squared distribution.

    Args:
        normal_vectors (np.ndarray): Representations of normal samples with shape (N, D).
        test_vectors (np.ndarray): Representations of test samples with shape (M, D).
        reg_eps (float): Regularization term added to the covariance matrix (default: 1e-3).
    
    Returns:
        p_values (np.ndarray): p-values for each test sample, with shape (M,).
    """
    # 1. Estimate the mean vector and covariance matrix from the given samples.
    mu = np.mean(normal_vectors, axis=0)
    cov = np.cov(normal_vectors, rowvar=False)
    cov += reg_eps * np.eye(cov.shape[0]) 
    cov_inv = np.linalg.inv(cov)

    # 2. Compute the squared Mahalanobis distance.
    delta = test_vectors - mu  # shape = (M, D)
    left = np.dot(delta, cov_inv)  # shape = (M, D)
    mahalanobis_squared = np.sum(left * delta, axis=1)  # shape = (M,)

    # 3. Transform the squared Mahalanobis distances into p-values under the chi-squared distribution.
    df = normal_vectors.shape[1] 
    p_values = 1 - chi2.cdf(mahalanobis_squared, df)

    return p_values

# The SFT samples are represented in the IB latent space of InfoRM with shape [num_samples, latent_dim].
sft_representation = ... 
# The RLHF samples are represented in the IB latent space of InfoRM with shape [num_samples, latent_dim].
rlhf_representation = ...
num_sample = sft_representation.shape[0]

# Concat
all_representation = np.concatenate((sft_representation, rlhf_representation), axis=0)  # shape: (2 * num_samples, latent_dim)

umap_model = UMAP(
    n_components=75, 
    n_neighbors=30, 
    min_dist=0.5,        
    metric='cosine', 
    random_state=0
)

all_representation_compressed = umap_model.fit_transform(all_representation)

all_representation_compressed = (all_representation_compressed - np.min(all_representation_compressed, 0)) / \
                                (np.max(all_representation_compressed, 0) - np.min(all_representation_compressed, 0))

sft_representation_compressed = all_representation_compressed[:num_sample, :]
rlhf_representation_compressed = all_representation_compressed[num_sample:, :]

mask = get_gaussian_mask(sft_representation, alpha=0.1, reg_eps=1e-6)

ib_representation_sft = sft_representation_compressed[mask]
ib_representation_rlhf = rlhf_representation_compressed[mask]
pvals = compute_hacking_pvalues(ib_representation_sft, ib_representation_rlhf)