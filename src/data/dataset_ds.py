import numpy as np
import torch
from torch.utils.data import Dataset

from tf_data_process import DataProcessorForPad


class DatasetMultiPad(Dataset):
    def __init__(self, *args, **kwargs):
        self.return_index = False
        self.return_batch_label = kwargs['return_batch_label']

        self.cell_type_map = kwargs['cell_type_map']
        self.cell_type_col = kwargs['cell_type_col']
        if self.return_batch_label:
            self.batch_label_map = kwargs['batch_label_map']
            self.batch_label_col = kwargs['batch_label_col']
        else:
            self.batch_label_map = None
            self.batch_label_col = None
        self.data_args = kwargs
        self.data_processor = DataProcessorForPad(**self.data_args)
        self.adata_list = args
        self.cumsum_lengths = np.cumsum([adata.shape[0] for adata in self.adata_list])

    def __len__(self):
        return self.cumsum_lengths[-1]

    def process_data(self, sparse_matrix, file_index):
        value_list, chromosome_list, hg38_start, hg38_end = \
            self.data_processor.process(
                value_data=sparse_matrix.toarray()[0].tolist(),
                chromosome=self.adata_list[file_index].var["#Chromosome"].tolist(),
                hg38_start=self.adata_list[file_index].var["hg38_Start"].tolist(),
                hg38_end=self.adata_list[file_index].var["hg38_End"].tolist()
            )
        return value_list, chromosome_list, hg38_start, hg38_end

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumsum_lengths, idx, side="right")
        if file_index == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cumsum_lengths[file_index - 1]
        x_idx = self.adata_list[file_index][row_idx].X
        value_list, chromosome_list, hg38_start, hg38_end = self.process_data(
            x_idx, file_index)
        cell_type = self.adata_list[file_index].obs.iloc[row_idx][self.cell_type_col]
        if cell_type not in self.cell_type_map:
            cell_type = 'Astrocyte 1'
        res = [
            torch.tensor(value_list),
            torch.tensor(chromosome_list),
            torch.tensor(hg38_start),
            torch.tensor(hg38_end),
            torch.tensor(self.cell_type_map[cell_type])
        ]
        if self.return_index:
            res.insert(0, torch.tensor(idx))
        if self.return_batch_label:
            batch_label = self.adata_list[file_index].obs.iloc[row_idx][self.batch_label_col]
            res.append(torch.tensor(self.batch_label_map[batch_label]))
        return res

    def __del__(self):
        # Properly close the backed file when the dataset is deleted
        [data.file.close() for data in self.adata_list]
