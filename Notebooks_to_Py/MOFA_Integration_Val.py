#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import prot as pt
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mofax as mofa
import scib

# In[3]:


#Data Loading and PreProcessing ( for FLU DATA ) 
adata = sc.read("./data/flu_vacc_CITEseq_combinedassay.h5ad")

adata.obs["celltype"] = adata.obs["celltype_joint"].astype(str).astype('category')
adata.var["gene_name"] = adata.var.index.tolist()
adata.var['feature_types'] = adata.var['features'].apply(lambda x: 'ADT' if '_PROT' in x else 'GEX')
adata.obs["str_batch"] = adata.obs['batch'].astype('category')
adata_protein = adata[:, adata.var.feature_types.isin(['ADT'])].copy()
adata_protein.var.index = ['p_' + i for i in adata_protein.var.index]
adata_rna = adata[:, adata.var.feature_types.isin(['GEX'])].copy()
data_is_raw = False
data_matrix = adata_protein.X.toarray() if issparse(adata_protein.X) else adata_protein.X
adata_protein.X = data_matrix

#Scaling the data to fix the Non-zero negatives
scaler = MinMaxScaler()
adata_protein.X = scaler.fit_transform(adata_protein.X)

#creating new modality to store protein expression
prot = adata_protein
prot

mdata = mu.MuData({'rna': adata_rna, 'protein': adata_protein})

# In[5]:


#PCA And UMAP (Clustering and Dimensionality)
sc.tl.pca(prot)
sc.pl.pca(prot, color = "celltype")
sc.pp.neighbors(prot)

sc.tl.umap(prot, random_state=1)
sc.pl.umap(prot, color = "celltype")

rna = adata_rna
rna

sc.pp.normalize_total(rna, target_sum=1e4)      #Uncomment for FLU DATA
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
sc.pl.highly_variable_genes(rna)
np.sum(rna.var.highly_variable)

sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, svd_solver='arpack')
sc.pl.pca(rna,color = "celltype")

sc.pl.pca_variance_ratio(rna, log=True)

sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
sc.tl.leiden(rna, resolution=.75)
sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)

sc.pl.umap(rna, color="celltype", frameon=False,
          title="RNA celltype annotation")


#MOFA RUN 
mu.pl.embedding(mdata, basis="protein:X_umap", color="protein:celltype")
mu.pp.intersect_obs(mdata)
mu.tl.louvain(mdata, resolution=[2, .1], random_state=1)

mu.pl.embedding(mdata, basis="rna:X_umap", color="louvain")
prot.var["highly_variable"] = True
mdata.update()

#Change the name of outfile arg to create new MOFA Model
mu.tl.mofa(mdata, outfile="models/H1N1_citeseq_v4.hdf5",    
           n_factors=30)              
mu.pl.mofa(mdata, color='louvain')

sc.pp.neighbors(mdata, use_rep="X_mofa")
sc.tl.umap(mdata, random_state=1)
mu.pl.umap(mdata, color=['rna:celltype'], frameon=False,
           title="UMAP(MOFA) embedding with RNA celltype annotation")

# In[ ]:


rcParams['figure.dpi'] = 150
model = mofa.mofa_model("models/H1N1_citeseq_v4.hdf5")   #load model
mofa.plot_weights(model, factors=range(3), n_features=10, sharex=False)
mdata
mdata.obsm["X_mofa_umap"] = mdata.obsm["X_umap"]


# WNN INTEGRATION ( OPTIONAL CHUNK )

# In[ ]:


# # Since subsetting was performed after calculating nearest neighbours,
# # we have to calculate them again for each modality.
# sc.pp.neighbors(mdata.mod['rna'])
# sc.pp.neighbors(mdata.mod['protein'])


# # Calculate weighted nearest neighbors
# mu.pp.neighbors(mdata, key_added='wnn')

# mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)

# print(mdata.obs.columns)

# #Compare modalities weights
# mu.pl.umap(mdata, color=['rna:mod_weight', 'protein:mod_weight'], cmap='RdBu')

# #Calculate neighbors and generate umaps
# sc.tl.leiden(mdata, resolution=1.0, neighbors_key='wnn', key_added='leiden_wnn')
# sc.pl.umap(mdata, color='leiden_wnn', legend_loc='on data')

# sc.pl.violin(mdata, groupby='leiden_wnn', keys='protein:mod_weight')

# mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]
# mdata.mod['rna'].obsm['X_wnn_umap'] = mdata.obsm['X_wnn_umap']


# In[ ]:


# #MOFA and WNN Umap Comparison
# mu.pl.embedding(mdata, basis="X_mofa_umap", frameon=False, title="MOFA\u2192UMAP", color="leiden_mapped")
# mu.pl.embedding(mdata, basis="X_wnn_umap", frameon=False, title="WNN\u2192UMAP", color="leiden_wnn")


# In[ ]:


# # Assuming 'rna' is your AnnData object
# sc.tl.leiden(rna, resolution=0.75, flavor='igraph', n_iterations=2, directed=False)


# In[ ]:


#preprocessing for Metric Collection
sc.tl.rank_genes_groups(rna, 'leiden', method='t-test_overestim_var')
rna.obs.celltype = rna.obs.celltype.astype("category")

sc.pp.neighbors(mdata, use_rep='X_mofa')
sc.tl.leiden(mdata, resolution=0.5)
sc.tl.umap(mdata)
sc.pl.umap(mdata, color=['leiden'])

#save labels as data frames and get mappings
celltype = pd.DataFrame(mdata.mod['rna'].obs['celltype_m_joint'])
leiden = pd.DataFrame(mdata.obs['leiden'])
merged_df = pd.merge(celltype, leiden, left_index=True, right_index=True)

mapping = merged_df[['leiden','celltype_m_joint']].set_index('leiden').to_dict()['celltype_m_joint']
leiden_annotations = merged_df.groupby('leiden')['celltype_m_joint'].agg(lambda x: x.value_counts().idxmax())
leiden_to_celltype = leiden_annotations.to_dict()
print("Leiden to Cell Type Mapping:", leiden_to_celltype)

#Assign mappings 
mdata.obs['leiden_mapped'] = mdata.obs['leiden'].map(leiden_to_celltype)
mdata.obs['leiden_mapped'].nunique()
sc.pl.umap(mdata, color='leiden_mapped')
mdata.mod['rna'].obs['leiden_mapped'] = mdata.obs['leiden_mapped']
mdata.mod['rna'].obsm['X_mofa_umap'] = mdata.obsm['X_mofa_umap']

#MOFA Evaluation Metrics
nmi = scib.metrics.nmi(mdata.mod['rna'], 'celltype', 'leiden_mapped')
ari = scib.metrics.ari(rna, 'celltype', 'leiden_mapped')
silhouette = scib.metrics.silhouette(rna, 'leiden_mapped', embed='X_mofa_umap')

print(f"NMI: {nmi}")
print(f"ARI: {ari}")
print(f"Silhouette Score: {silhouette}")
print(f"avg_bio: {(nmi + ari + silhouette) / 3}")
