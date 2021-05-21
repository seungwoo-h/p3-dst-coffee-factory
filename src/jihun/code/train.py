import argparse
import json
import os
import random
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torchcontrib.optim import SWA
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup, ElectraModel, ElectraTokenizer, BertConfig

from data_utils import (WOSDataset, get_examples_from_dialogues, load_dataset,
                        set_seed)
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference
from model import TRADE, masked_cross_entropy_for_value
from preprocessor import TRADEPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlm_pretrain(loader, n_epochs):
    model.train()
    for step, batch in enumerate(loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [b.to(device) if not isinstance(b, list) else b for b in batch]

        logits, labels = model.forward_pretrain(input_ids, tokenizer)
        loss = loss_fnc_pretrain(logits.view(-1, config.vocab_size), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print('[%d/%d] [%d/%d] %f' % (epoch, n_epochs, step, len(loader), loss.item()))

if __name__ == "__main__":
    wandb.init(project="DST", reinit=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="results3")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="kykim/bert-kor-base",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=None,
                        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()
    wandb.config.update(args)
    # args.data_dir = os.environ['SM_CHANNEL_TRAIN']
    # args.model_dir = os.environ['SM_MODEL_DIR']

    # random seed 고정
    set_seed(args.random_seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Define Preprocessor
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    processor = TRADEPreprocessor(slot_meta, tokenizer)
    args.vocab_size = len(tokenizer)
    print("vocab size = ", end="")
    print(args.vocab_size)
    args.n_gate = len(processor.gating2id) # gating 갯수 "none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)
    
    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )
    
    # Model 선언
    config = BertConfig.from_pretrained("kykim/bert-kor-base")
    config.model_name_or_path = "kykim/bert-kor-base"
    config.n_gate = len(processor.gating2id)
    config.proj_dim = None
    model = TRADE(config, tokenized_slot_meta, slot_meta)
    # model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화
    print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
    model.to(device)
    print("Model is initialized")
    wandb.watch(model)
    
    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# dev:", len(dev_data))
    
    # Optimizer 및 Scheduler 선언
    n_epochs = args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    base_opt = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    teacher_forcing = 0.5
    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating
    loss_fnc_pretrain = nn.CrossEntropyLoss()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

        
    json.dump(
        vars(args),
        open(f"{args.model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    MLM_PRE = True
    n_pretrain_epochs = 3

    if MLM_PRE:
        for epoch in range(n_pretrain_epochs):
            mlm_pretrain(train_loader, n_pretrain_epochs)
    
    MLM_DURING = True
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ]
            
            # teacher forcing
            if (
                args.teacher_forcing_ratio > 0.0
                and random.random() < args.teacher_forcing_ratio
            ):
                tf = target_ids
            else:
                tf = None

            all_point_outputs, all_gate_outputs = model(
                input_ids, segment_ids, input_masks, target_ids.size(-1), tf
            )
            
            # generation loss
            loss_1 = loss_fnc_1(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1),
                tokenizer.pad_token_id,
            )
            
            # gating loss
            loss_2 = loss_fnc_2(
                all_gate_outputs.contiguous().view(-1, args.n_gate),
                gating_ids.contiguous().view(-1),
            )
            loss = loss_1 + loss_2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )
                wandb.log({"Train Loss": loss.item(),
                          "Train Gen": loss_1.item(),
                          "Train Gate": loss_2.item()})
        optimizer.swap_swa_sgd()
        if MLM_DURING:
            mlm_pretrain(train_loader, n_epochs)

        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
