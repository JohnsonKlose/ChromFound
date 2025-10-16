import argparse
import yaml
import scanpy as sc
from atac_preprocess import quality_control
from atac_preprocess import deepen_atac_data
from atac_preprocess import tfidf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to config file")
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    data_args = config["data_args"]
    adata = sc.read_h5ad(data_args.get('atac_file_path'))
    # adata.obs['celltype'] = adata.obs['Celltype_fig1']
    if data_args.get("quality_control", False):
        print(f"Before quality control: {adata.shape}")
        adata = quality_control(
            adata,
            min_features=data_args.get("min_features"),
            max_features=data_args.get("max_features"),
            min_percent=data_args.get("min_percent"),
            cell_type_col=data_args.get("cell_type_col", "cell type")
        )
        print(f"After quality control: {adata.shape}")

    if data_args.get("tfidf", False):
        print(f"Before tfidf: {adata.shape}")
        adata.X = tfidf(adata.X) * 1e4

    if data_args.get("deepen_atac", False):
        adata = deepen_atac_data(adata, num_cell_merge=10)
        print(f"After deepen: {adata.shape}")
    if data_args.get("normalize", False):
        sc.pp.normalize_total(adata)
    if data_args.get("log_transform", False):
        sc.pp.log1p(adata)

    output_file_name = f"{data_args.get('atac_file_path')}".replace(".h5ad", "")
    if data_args.get("quality_control", False):
        output_file_name = f"{output_file_name}_quality_control"
    if data_args.get("deepen_atac", False):
        output_file_name = f"{output_file_name}_deepen"
    if data_args.get("normalize", False):
        output_file_name = f"{output_file_name}_normalized"
    if data_args.get("log_transform", False):
        output_file_name = f"{output_file_name}_log"

    output_file_name = f"{output_file_name}.h5ad"
    adata.write_h5ad(output_file_name)
