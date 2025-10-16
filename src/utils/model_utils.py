import yaml


class ModelUtils:
    @classmethod
    def get_chromosome_vocab(cls, file_path):
        with open(file_path) as file:
            chromosome_vocab = yaml.safe_load(file)
            chromosome_vocab = {chr_: idx for idx, chr_ in enumerate(chromosome_vocab["chromosome"])}
        return chromosome_vocab
