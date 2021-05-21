import json
from collections import defaultdict
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import copy
from re import findall

def recover_state(pred_slot):
        states = []
        for s, v in zip(slot_meta, pred_slot):
            if v != 'none':
                states.append(f'{s}-{v}')
        return states

def make_every_slot(datas, weights):
    acc = []
    for slot in slot_meta:
        slot_cnt = defaultdict(float)
        for idx, (data, weight) in enumerate(zip(datas, weights)):
            if len(ontology[slot]) >= 12 and idx % 2 == 1:
                continue
            d = {'-'.join(v.split('-')[:2]):v.split('-')[2] for v in data}
            slot_cnt[d.get(slot, 'none')] += weight
            
        maxval = max(slot_cnt.values())
        res = [k for k, v in slot_cnt.items() if v == maxval]
        acc.append(res[0])
    
    return recover_state(acc)

def get_cls_token(sent_A):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=512
    )
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(    # **tokenized_sent
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs[1].detach().cpu().numpy()
    return logits

slot_meta = json.load(open("/opt/ml/input/data/train_dataset/slot_meta.json"))
ontology = json.load(open("/opt/ml/input/data/train_dataset/ontology.json"))


MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

ontology_list = []
for key in ontology:
    if "이름" in key or "출발지" in key or "도착지" in key:
        ontology_list.extend(ontology[key])
    
dataset_cls_hidden = []
for q in ontology_list:
    q_cls = get_cls_token(q)
    dataset_cls_hidden.append(q_cls)
dataset_cls_hidden = np.array(dataset_cls_hidden).squeeze(axis=1)

sumbt63 = json.load(open('/opt/ml/code/output/output_sumbt_06344.json'))
trade734 = json.load(open('/opt/ml/code/output/output_trade_07345.json'))
sumbt62 = json.load(open('/opt/ml/code/output/prediction_sumbt_06295.json'))
trade738 = json.load(open('/opt/ml/code/output/predictions_trade_07382.json'))
trade710 = json.load(open('/opt/ml/code/output/predictions_0.7101.json'))
predicts = {}
for i, dlg in enumerate(zip(trade734, sumbt63, trade738, sumbt62, trade710)):
    predicts[dlg[0]] = make_every_slot([trade734[dlg[1]], sumbt63[dlg[0]], trade738[dlg[3]], sumbt62[dlg[2]], trade710[dlg[4]]], [0.7345,0.6344, 0.7382, 0.6295, 0.7101])

for predict in predicts:
    data = predicts[predict]
    for v in data:
        sl = v.split('-')[:2]
        if "이름" in sl or "출발지" in sl or "도착지" in sl:
            val = v.split('-')[2]
            query_cls_hidden = get_cls_token(val)
            cos_sim = cosine_similarity(query_cls_hidden, dataset_cls_hidden)
            top_question = np.argmax(cos_sim)
            if ontology_list[top_question] != val and max(cos_sim[0]) >= 0.98:
                print(cos_sim.shape)
                print(max(cos_sim[0]))
                print(ontology_list[top_question])
                print(val)
                v = v.replace(v.split('-')[2], ontology_list[top_question])


json.dump(predicts, open('/opt/ml/code/output/predictions.csv', 'w'), indent=2, ensure_ascii=False)