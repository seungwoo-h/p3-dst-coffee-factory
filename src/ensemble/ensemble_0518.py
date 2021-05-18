from collections import defaultdict

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
        for data, weight in zip(datas, weights):
            d = {'-'.join(v.split('-')[:2]):v.split('-')[2] for v in data}
            slot_cnt[d.get(slot, 'none')] += weight
        maxval = max(slot_cnt.values())
        res = [k for k, v in slot_cnt.items() if v == maxval]
        acc.append(res[0])
    
    return recover_state(acc)

sumbt63 = json.load(open('/opt/ml/input/result/output_sumbt_06344.json'))
trade734 = json.load(open('/opt/ml/input/result/output_trade_07345.json'))
sumbt62 = json.load(open('/opt/ml/input/result/prediction_sumbt_06295.json'))
trade738 = json.load(open('/opt/ml/input/result/predictions_trade_07382.json'))

predicts = {}
for i, dlg in enumerate(zip(sumbt63, trade734, sumbt62, trade738)):
    predicts[dlg[0]] = make_every_slot([sumbt63[dlg[0]], trade734[dlg[1]], sumbt62[dlg[2]], trade738[dlg[3]]], [0.6344, 0.7345, 0.6295, 0.7382])

json.dump(predicts, open('predictions.csv', 'w'), indent=2, ensure_ascii=False) 
