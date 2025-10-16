class DataProcessorForPad:
    def __init__(self, **kwargs):
        self.chromosome_vocab = kwargs.get("chromosome_vocab")
        self.max_length = kwargs.get("max_length", 1024)
        self.add_cls = kwargs.get('add_cls', False)
        self.padding_value = kwargs.get('padding_value', -1)
        if self.add_cls:
            self.max_length -= 1

    def process(self, value_data, chromosome, hg38_start, hg38_end):
        chromosome_list = [self.chromosome_vocab[chr_] for chr_ in chromosome]
        # padding
        if len(value_data) >= self.max_length:
            value_data = value_data[:self.max_length]
            chromosome_list = chromosome_list[:self.max_length]
            hg38_start = hg38_start[:self.max_length]
            hg38_end = hg38_end[:self.max_length]
        else:
            value_data.extend([self.padding_value] * (self.max_length - len(value_data)))
            chromosome_list.extend([self.chromosome_vocab["pad"]] * (self.max_length - len(chromosome_list)))
            hg38_start.extend([0] * (self.max_length - len(hg38_start)))
            hg38_end.extend([0] * (self.max_length - len(hg38_end)))
        if self.add_cls:
            value_data.insert(0, 0)
            chromosome_list.insert(0, self.chromosome_vocab["pad"])
            hg38_start.insert(0, 0)
            hg38_end.insert(0, 0)
        return value_data, chromosome_list, hg38_start, hg38_end
