def get_file_to_nnumber(filepath):
    valmapping = {}
    with open(filepath) as f:
        for line in f:
            info = line.split('	')
            valmapping[info[0]] = info[1]  # val_XXX.JPEG: nnumber
        return valmapping


def get_nnumber_to_name(filepath):
    mapping = {}
    with open(filepath) as f:
        for line in f:
            (nnumber, name) = line.split('	')
            mapping[nnumber] = name
        return mapping


def get_labels(labels, file_names, valmapping, nnumber_to_idx):
    for i in range(len(labels)):
        nnumber = valmapping[file_names[i]]
        labels[i] = nnumber_to_idx[nnumber]
    return labels