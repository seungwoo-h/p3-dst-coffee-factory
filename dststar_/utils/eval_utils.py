import torch
import numpy as np
import json
import time
import os
from copy import deepcopy
import torch
import time


def convert_millisecond_to_str(millisecond: float, include_decimal=False):
    r'''Convert time to str
    Args:
        millisecond (float): elapsed time recorded by torch.cuda.Event
        include_decimal (bool): whether include decimal points to second
    '''
    second, decimal = divmod(int(millisecond), 1000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    decimal = str(decimal).rjust(3, '0')

    time_str = f'{minute:02d}:{second:02d}'
    if hour > 0:
        time_str = f'{hour:02d}:' + time_str

    if include_decimal:
        time_str += f'.{decimal}'
    
    return time_str


def model_evaluation(model, test_data, tokenizer, slot_meta, label_list, epoch, is_gt_p_state=False, submit=False):
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    
    final_count = 0
    loss = 0.
    joint_acc = 0.
    joint_turn_acc = 0.
    final_joint_acc = 0. # only consider the last turn in each dialogue
    slot_acc = np.array([0.] * len(slot_meta))
    final_slot_acc = np.array([0.] * len(slot_meta))

    results = {}
    last_dialogue_state = {}
    wall_times = [] 
    for di, i in enumerate(test_data):
        if i.turn_id == 0 or is_gt_p_state:
            last_dialogue_state = deepcopy(i.gold_last_state)
            
        i.last_dialogue_state = deepcopy(last_dialogue_state)
        i.make_instance(tokenizer)

        input_ids = torch.LongTensor([i.input_id]).to(model.device)
        input_mask = torch.LongTensor([i.input_mask]).to(model.device)
        segment_ids = torch.LongTensor([i.segment_id]).to(model.device)
        label_ids = torch.LongTensor([i.label_ids]).to(model.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            t_loss, _, t_acc, t_acc_slot, t_pred_slot = model(input_ids=input_ids, attention_mask=input_mask,
                                                              token_type_ids=segment_ids, labels=label_ids,
                                                              eval_type="test")
            loss += t_loss.item()
            joint_acc += t_acc
            slot_acc += t_acc_slot
            if i.is_last_turn:
                final_count += 1
                final_joint_acc += t_acc
                final_slot_acc += t_acc_slot
            
        end = time.perf_counter()
        wall_times.append(end - start)
        
        ss = {}
        t_turn_label = []
        dialogue_state = {}
        for s, slot in enumerate(slot_meta):
            v = label_list[s][t_pred_slot[0, s].item()]
            if v != last_dialogue_state[slot]:
                t_turn_label.append(slot + "-" + v)
            last_dialogue_state[slot] = v
            vv = label_list[s][i.label_ids[s]]
            if v == vv:
                t_turn_label.append(slot + '-' + vv)
            ss[slot] = {}
            ss[slot]["pred"] = v
            ss[slot]["gt"] = vv
        
        if set(t_turn_label) == set(i.turn_label):
            joint_turn_acc += 1

        key = str(i.dialogue_id) + '-' + str(i.turn_id)
        results[key] = [f"{s}-{v}" for s, v in map(lambda x: x.rsplit('-', 1), t_turn_label)]

        ender.record()
        torch.cuda.synchronize()
        dur = starter.elapsed_time(ender)
        dur = convert_millisecond_to_str(dur)
        print(f"\r({dur}) Evaluating {(di + 1) / len(test_data) * 100:.1f}%", end="")
    
    print()

    if not submit:    
        loss = loss / len(test_data)
        joint_acc_score = joint_acc / len(test_data)
        joint_turn_acc_score = joint_turn_acc / len(test_data)
        slot_acc_score = slot_acc / len(test_data)
        final_joint_acc_score = final_joint_acc / final_count
        final_slot_acc_score = final_slot_acc / final_count
        
        latency = np.mean(wall_times) * 1000 # ms
        
        print("------------------------------")
        print('is_gt_p_state: %s' % (str(is_gt_p_state)))
        print("Epoch %d loss : " % epoch, loss)
        print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
        print("Epoch %d joint turn accuracy : " % epoch, joint_turn_acc_score)
        print("Epoch %d slot accuracy : " % epoch, np.mean(slot_acc_score))
        print("Final Joint Accuracy : ", final_joint_acc_score)
        print("Final slot Accuracy : ", np.mean(final_slot_acc_score))
        print("Latency Per Prediction : %f ms" % latency)
        print("-----------------------------\n")
        # if not os.path.exists("pred"):
        #     os.makedirs("pred")
        json.dump(results, open('/opt/ml/output/submissions/eval_%d.json' % epoch, 'w'), indent=2, ensure_ascii=False)
        scores = {'epoch': epoch, 'loss': loss, 'joint_acc': joint_acc_score, 'joint_turn_acc': joint_turn_acc_score, 'slot_acc': slot_acc_score, 'ave_slot_acc': np.mean(slot_acc_score), 'final_joint_acc': final_joint_acc_score, 'final_slot_acc': final_slot_acc_score, 'final_ave_slot_acc': np.mean(final_slot_acc_score)}
        return scores

    else:
        json.dump(results, open(f'/opt/ml/output/submissions/sub_star{time.time():.1f}.csv', 'w'), indent=2, ensure_ascii=False)

