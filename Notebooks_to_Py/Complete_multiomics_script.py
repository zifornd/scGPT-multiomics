#!/usr/bin/env python
# coding: utf-8

import pickle
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
#from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
from anndata import AnnData
import scanpy as sc
#import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing

sys.path.insert(0, "../")
from scgpt import prepare_data, prepare_dataloader, define_wandb_metrcis, evaluate, eval_testdata, train
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.model import MultiOmicTransformerModel

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# Hyper parmeter
hyperparameter_defaults = dict(
    task = 'multiomic',
    seed=42,
    dataset_name="Flu", # Dataset name
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model="./save/scGPT_human", # Path to pre-trained model
    freeze = False, #freeze
    GEP=True, # Gene expression modelling
    GEPC=True, # Gene expression modelling for cell objective
    CLS=False,
    ESC=False,
    DAR = True, # DAR objective weight for batch correction
    DSBN = False,  # Domain-spec batchnorm,
    mask_ratio=0.4, # Default mask ratio
    explicit_zero_prob = False,  # whether explicit bernoulli for zeros
    ecs_thres=0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    use_batch_labels = True,
    use_mod = True,
    per_seq_batch_sample = False,
    epochs= 25,   #25 Default number of epochs for fine-tuning         #MODIFY
    input_layer_key = "X_binned", # Default expression value binning in data pre-processing
    n_bins= 51,    # Default number of bins for value binning in data pre-processing
    n_hvg = 1200,  # Default number of highly variable genes
    n_hvp = 4000,
    max_seq_len = 4001, # # Default n_hvg+1
    lr=1e-4, #1e-3, # Default learning rate for fine-tuning                   #MODIFY
    batch_size= 48, #16, # Default batch size for fine-tuning               #MODIFY
    layer_size=512,
    nlayers=4,
    nhead=8, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2, # Default dropout rate during model fine-tuning
    schedule_ratio=0.95,  # Default rate for learning rate decay
    save_eval_interval=5, # Default model evaluation interval
    log_interval=100,  # Default log interval
    fast_transformer=True, # Default setting
    pre_norm=False, # Default setting
    amp=True,  # Default setting: Automatic Mixed Precision
    pad_token = "<pad>",
    mask_value = -1,
    pad_value = -2,
    include_zero_gene = False,
)

# Manually set the configuration
config = hyperparameter_defaults
print(config)

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed using the value from the config
set_seed(config['seed'])

# Settings for input and preprocessing
special_tokens = [config['pad_token'], "<cls>", "<eoc>"]

# Dataset
dataset_name = config['dataset_name']
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# Data ingestion
if dataset_name == 'Flu':
    adata = sc.read("./data/flu_vacc_CITEseq_combinedassay.h5ad")
    adata = adata[:10000,:]
    adata.obs["celltype"] = adata.obs["celltype_joint"].astype(str).astype('category')
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var['feature_types'] = adata.var['features'].apply(lambda x: 'ADT' if '_PROT' in x else 'GEX')
    le = preprocessing.LabelEncoder()
    encoded_batch = le.fit_transform(adata.obs['batch'].values)
    adata.obs["batch_id"] =  encoded_batch
    adata.obs["str_batch"] = adata.obs["batch_id"].astype('category')
    adata_protein = adata[:, adata.var.feature_types.isin(['ADT'])].copy()
    adata_protein.var.index = ['p_' + i for i in adata_protein.var.index]
    adata = adata[:, adata.var.feature_types.isin(['GEX'])].copy()
    data_is_raw = False
    data_matrix = adata_protein.X.toarray() if issparse(adata_protein.X) else adata_protein.X
    adata_protein.X = data_matrix

#Check for Negative Values in Array ( Comment out next block for Scaling if Negative Values = 0 )

# data_matrix = adata.X.toarray() if issparse(adata.X) else adata.X
negative_values = data_matrix[data_matrix < 0]
print(f"Negative values: {negative_values}")
print(len(negative_values))

#Scaling the data to fix the Non-zero negatives
scaler = MinMaxScaler()
adata_protein.X = scaler.fit_transform(adata_protein.X)

if config['use_mod']:
    gene_rna_df = pd.DataFrame(index = adata.var.index.tolist())
    gene_rna_df['mod'] = 'RNA'
    gene_protein_df = pd.DataFrame(index = adata_protein.var.index.tolist())
    gene_protein_df['mod'] = 'Protein'
    gene_loc_df = pd.concat([gene_rna_df, gene_protein_df])
    gene_loc_df['mod'] = gene_loc_df['mod'].astype('category')

#display((gene_loc_df['mod']))

data_is_raw, logger, save_dir, dataset_name


#Cross-check gene set with the pre-trained model 
if config['load_model'] is not None:
    model_dir = Path(config['load_model'])
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    old_vocab = vocab

# Set model parameters
embsize = config['layer_size']
nhead = config['nhead']
nlayers = config['nlayers']
d_hid = config['layer_size']

# Pre-processing
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=1,  # step 1
    filter_cell_by_counts=1,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=config['n_hvg'],  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config['n_bins'],  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key=None)

preprocessor_protein = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=0,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor=None,
    binning=config['n_bins'],  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor_protein(adata_protein, batch_key=None)

data_combined = np.concatenate([adata.layers["X_binned"], adata_protein.layers["X_binned"]], axis=1)
adata = AnnData(
    X=data_combined,
    obs=adata.obs,
    var=pd.DataFrame(index=adata.var_names.tolist() + adata_protein.var_names.tolist()),
    layers={"X_binned": data_combined,}
)
adata.var["gene_name"] = adata.var.index.tolist()

if config['per_seq_batch_sample']:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()


all_counts = (
    adata.layers[config['input_layer_key']].A
    if issparse(adata.layers[config['input_layer_key']])
    else adata.layers[config['input_layer_key']]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

if config['use_mod']:
    mod_type = np.array([gene_loc_df.loc[g, 'mod'] for g in genes])
    vocab_mod = Vocab(VocabPybind(np.unique(gene_loc_df['mod']).tolist() + special_tokens, None))
    vocab_mod.set_default_index(vocab_mod["<pad>"])
    mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)
    ntokens_mod = len(vocab_mod)
    print(mod_type)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

num_of_non_zero_genes = [
    np.count_nonzero(train_data[i]) for i in range(train_data.shape[0])
]
print(f"max num of non_zero genes: {np.max(num_of_non_zero_genes)}")
print(f"min num of non_zero genes: {np.min(num_of_non_zero_genes)}")
print(f"average num of non_zero genes: {np.mean(num_of_non_zero_genes)}")
print(
    f"99% quantile num of non_zero genes: {np.quantile(num_of_non_zero_genes, 0.99)}"
)
print(f"max original values: {np.max(train_data)}")
print(
    f"average original non_zero values: {np.mean(train_data[np.nonzero(train_data)])}"
)
print(
    f"99% quantile original non_zero values: {np.quantile(train_data[np.nonzero(train_data)], 0.99)}"
)
print(f"num of celltypes: {num_types}")


if config['load_model'] is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
else:
    pretrained_genes = [g for g in genes + special_tokens if g in old_vocab]
    new_genes = [g for g in genes + special_tokens if g not in old_vocab]
    gene_ids_pretrained = np.array(old_vocab(pretrained_genes), dtype=int)
    # https://discuss.pytorch.org/t/expand-an-existing-embedding-and-linear-layer-nan-loss-value/55670/2
    # Retrieve pretrained weights
    vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

# Tokenization
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config['max_seq_len'],
    vocab=vocab,
    pad_token=config['pad_token'],
    pad_value=config['pad_value'],
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=config['include_zero_gene'],
    mod_type=mod_type if config['use_mod'] else None,
    vocab_mod=vocab_mod if config['use_mod'] else None,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=config['max_seq_len'],
    vocab=vocab,
    pad_token=config['pad_token'],
    pad_value=config['pad_value'],
    append_cls=True,
    include_zero_gene=config['include_zero_gene'],
    mod_type=mod_type if config['use_mod'] else None,
    vocab_mod=vocab_mod if config['use_mod'] else None,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = torch.load(model_file)
ntokens = len(vocab)  # size of vocabulary
model = MultiOmicTransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid, 
    nlayers,
    vocab=vocab,
    dropout=config['dropout'],
    pad_token=config['pad_token'],
    pad_value=config['pad_value'],
    do_mvc=config['GEPC'],
    do_dab=config['DAR'],
    use_batch_labels=config['use_batch_labels'],
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config['DSBN'],
    n_input_bins=config['n_bins'],
    ecs_threshold=config['ecs_thres'],
    explicit_zero_prob=config['explicit_zero_prob'],
    use_fast_transformer=config['fast_transformer'],
    pre_norm=config['pre_norm'],
    use_mod=config['use_mod'],
    ntokens_mod=ntokens_mod if config['use_mod'] else None,
    vocab_mod=vocab_mod if config['use_mod'] else None,
)

with torch.no_grad():
    pretrained_emb_weights = model_dict['encoder.embedding.weight'][gene_ids_pretrained, :]
    model.encoder.embedding.weight.data[:len(pretrained_genes), :] = pretrained_emb_weights
    model.encoder.enc_norm.weight.data = model_dict['encoder.enc_norm.weight']
ntokens = len(vocab)

model.to(device)
#wandb.watch(model)
print(model)

if config['GEP'] and config['GEPC']:
    criterion_gep_gepc = masked_mse_loss
if config['CLS']:
    criterion_cls = nn.CrossEntropyLoss()
if config['DAR']:
    criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config['lr'], eps=1e-4 if config['amp'] else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config['schedule_ratio'])
scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])


#FineTuning
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
batch_first = True

for epoch in range(1, config['epochs'] + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(
        tokenized_train=tokenized_train, 
        tokenized_valid=tokenized_valid, 
        train_batch_labels=train_batch_labels,
        valid_batch_labels=valid_batch_labels,
        config=config,
        epoch=epoch,
        sort_seq_batch=config['per_seq_batch_sample'])
    
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config['batch_size'],
        shuffle=True,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config['per_seq_batch_sample']
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config['batch_size'],
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config['per_seq_batch_sample']
    )

    if config['do_train']:
        train(
            model=model,
            loader=train_loader,
            vocab=vocab,
            criterion_gep_gepc=criterion_gep_gepc if config['GEP'] and config['GEPC'] else None,
            criterion_dab=criterion_dab if config['DAR'] else None,
            criterion_cls=criterion_cls if config['CLS'] else None,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
        )
    val_loss = evaluate(
        model=model,
        loader=valid_loader,
        vocab=vocab,
        criterion_gep_gepc=criterion_gep_gepc if config['GEP'] and config['GEPC'] else None,
        criterion_dab=criterion_dab if config['DAR'] else None,
        criterion_cls=criterion_cls if config['CLS'] else None,
        device=device,
        config=config,
        epoch=epoch
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss {val_loss:5.4f} | "
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config['save_eval_interval'] == 0 or epoch == config['epochs']:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        # eval on testdata
        results = eval_testdata(
            model = best_model,
            adata_t = adata_sorted if config['per_seq_batch_sample'] else adata,
            gene_ids = gene_ids,
            vocab = vocab,
            config = config,
            logger = logger,
            include_types=["cls"],
        )
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
        )

        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        # metrics_to_log["test/batch_umap"] = wandb.Image(
        #     str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
        #     caption=f"celltype avg_bio epoch {best_model_epoch}",
        # )

        # metrics_to_log["test/celltype_umap"] = wandb.Image(
        #     str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
        #     caption=f"celltype avg_bio epoch {best_model_epoch}",
        # )
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        
    scheduler.step()

metrics_to_log

# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")

