import json
from data_utils import *

def merge_labels(data, labels):
    for d in data:
        idx = d['dialogue_idx']
        _c = 0
        for i_ in range(len(d['dialogue'])):
            if d['dialogue'][i_]['role'] != 'user':
                continue
            idx_ = f'{idx}-{_c}'
            label = labels[idx_]
            d['dialogue'][i_]['state'] = label
            _c += 1
    return data

if __name__ == '__main__':
    set_seed(42)

    train_data_file = "/opt/ml/input/data/train_dataset/train_dials.json"
    slot_meta = json.load(open("/opt/ml/input/data/train_dataset/slot_meta.json"))
    ontology = json.load(open("/opt/ml/input/data/train_dataset/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)
    test_data = json.load(open("/opt/ml/input/data/eval_dataset/eval_dials.json"))
    test_labels = json.load(open("/opt/ml/input/data/eval_dataset/eval_pseudo_labels.json"))

    dev_data = merge_labels(dev_data, dev_labels)
    test_data = merge_labels(test_data, test_labels)

    dev_data.extend(test_data)

    json.dump(
        dev_data,
        open("./additional_training_data.json", "w"),
        indent=2,
        ensure_ascii=False,
    )