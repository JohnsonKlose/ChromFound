import episcanpy.api as epi
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import sklearn
from scipy import sparse
from statsmodels.distributions.empirical_distribution import ECDF


def quality_control(
        adata_atac,
        min_features=1000,
        max_features=60000,
        min_percent=0.05,
        min_cells=None,
        cell_type_col='cell type',
):
    epi.pp.filter_cells(adata_atac, min_features=min_features)
    epi.pp.filter_cells(adata_atac, max_features=max_features)
    if min_percent is not None:
        by = adata_atac.obs[cell_type_col]
        agg_idx = pd.Index(by.cat.categories) if isinstance(by, pd.CategoricalDtype) else pd.Index(np.unique(by))
        agg_sum = sparse.coo_matrix(
            (np.ones(adata_atac.shape[0]), (agg_idx.get_indexer(by), np.arange(adata_atac.shape[0])))
        ).tocsr()
        # Ensure adata_atac.X is a sparse matrix
        if not scipy.sparse.issparse(adata_atac.X):
            adata_atac.X = scipy.sparse.csr_matrix(adata_atac.X)
        sum_x = agg_sum @ (adata_atac.X != 0)
        df_percent = pd.DataFrame(
            sum_x.toarray(), index=agg_idx, columns=adata_atac.var.index
        ) / adata_atac.obs.value_counts(cell_type_col).loc[agg_idx].to_numpy()[:, np.newaxis]
        df_percent_max = np.max(df_percent, axis=0)
        sel_peaks = df_percent.columns[df_percent_max > min_percent]
        adata_atac = adata_atac[:, sel_peaks]
    elif min_cells is not None:
        epi.pp.filter_features(adata_atac, min_cells=min_cells)
    return adata_atac


def tfidf(x):
    idf = x.shape[0] / (x.sum(axis=0) + 1e-6)
    if sparse.issparse(x):
        tf = x.multiply(1 / (x.sum(axis=1) + 1e-6))
        return tf.multiply(idf)
    else:
        tf = x / (x.sum(axis=1, keepdims=True) + 1e-6)
        return tf * idf


def lsi(
        adata,
        n_components=20,
        use_top_features=False,
        min_cutoff=0.05,
        **kwargs
):
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior

    adata_use = adata.copy()
    if use_top_features:
        adata_use.var['featurecounts'] = np.array(np.sum(adata_use.X, axis=0))[0]
        df_var = adata_use.var.sort_values(by='featurecounts')
        ecdf = ECDF(df_var['featurecounts'])
        df_var['percentile'] = ecdf(df_var['featurecounts'])
        df_var["selected_feature"] = (df_var['percentile'] > min_cutoff)
        adata_use.var = df_var.loc[adata_use.var.index, :]

    # factor_size = int(np.median(np.array(np.sum(adata_use.X, axis=1))))
    x_norm = np.log1p(tfidf(adata_use.X) * 1e4)
    if use_top_features:
        x_norm = x_norm.toarray()[:, adata_use.var["selected_feature"]]
    else:
        x_norm = x_norm.toarray()
    svd = sklearn.decomposition.TruncatedSVD(n_components=n_components, algorithm='arpack')
    X_lsi = svd.fit_transform(x_norm)
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


def deepen_atac_data(adata, num_pc=50, num_cell_merge=10):
    adata_atac_sample_cluster = adata.copy()
    lsi(adata_atac_sample_cluster, n_components=num_pc)
    adata_atac_sample_cluster.obsm["X_lsi"] = adata_atac_sample_cluster.obsm["X_lsi"][:, 1:]
    sc.pp.neighbors(
        adata_atac_sample_cluster,
        use_rep="X_lsi",
        metric="cosine",
        n_neighbors=int(num_cell_merge),
        n_pcs=num_pc-1
    )

    list_atac_index = []
    list_neigh_index = []
    for cell_atac in list(adata_atac_sample_cluster.obs.index):
        cell_atac = [cell_atac]
        cell_atac_index = np.where(adata_atac_sample_cluster.obs.index == cell_atac[0])[0]
        cell_neighbor_idx = np.nonzero(adata_atac_sample_cluster.obsp['connectivities'].getcol(cell_atac_index).toarray())[0]
        if num_cell_merge >= len(cell_neighbor_idx):
            cell_sample_atac = np.hstack([cell_atac_index, cell_neighbor_idx])
        else:
            cell_sample_atac = np.hstack([
                cell_atac_index, np.random.choice(cell_neighbor_idx, num_cell_merge, replace=False)
            ])
        list_atac_index.extend([cell_atac_index[0] for _ in range(len(cell_sample_atac))])
        list_neigh_index.append(cell_sample_atac)

    agg_sum = sparse.coo_matrix((
        np.ones(len(list_atac_index)), (np.array(list_atac_index), np.hstack(list_neigh_index))
    )).tocsr()
    array_atac = agg_sum @ adata.X

    # self.adata = self.adata.copy()
    adata.X = None
    adata.X = array_atac
    return adata


def chr_map_int(x):
    if x == "X" or x == "x":
        return 23
    elif x == "Y" or x == "y":
        return 24
    return int(x)
