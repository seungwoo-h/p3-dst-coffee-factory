from data_utils import *
from eval_utils import *
from collections import Counter
from nltk.util import ngrams

data_dir = "/opt/ml/input/data"
train_dir = data_dir+"/train_dataset"


datas = json.load(open(train_dir+"/train_dials.json"))
domains = dict()
for idx, data in enumerate(datas):
    print(f"idx {idx}, {data['domains']}")
    for domain in data['domains']:
        if domain not in domains:
            domains[domain] = 1
        else :
            domains[domain] += 1
print(domains)


slot_meta = count_slot_meta(datas)
print(slot_meta)

examples = get_examples_from_dialogues(datas)
value_meta = dict()
for example in examples:
    dic = convert_state_dict(example.label)
    for key, value in dic.items():
        if key not in value_meta.keys():
            value_meta[key] = [value]
        elif value not in value_meta[key]:
            value_meta[key].append(value)
            
value_meta_sorted_list = sorted(value_meta)

for key in value_meta_sorted_list:
    print(f"key : {key}, value : {value_meta[key]}")
    print()

n_gram = 2
bigram_meta = dict()
for idx, data in enumerate(datas):
    for dialogue in data['dialogue']:
        if dialogue['role'] == "user":.
            info = Counter(ngrams(dialogue['text'].split(), n_gram))
            for key in info.keys():
                for domain in data['domains']:    
                    if domain not in bigram_meta:
                        bigram_meta[domain] = {key:info[key]}
                    else :
                        if key in bigram_meta[domain].keys():
                            bigram_meta[domain][key] += info[key]
                        else :
                            bigram_meta[domain][key] = info[key]

for domain in bigram_meta.keys():
    print(domain)
    print(bigram_meta[domain])
    print()