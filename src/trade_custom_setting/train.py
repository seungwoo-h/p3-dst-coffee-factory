import argparse
import json
import os
import random
import numpy as np
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from madgrad import MADGRAD

from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tokenization_hanbert import HanBertTokenizer

from data_utils import *

from model import TRADE, masked_cross_entropy_for_value
from preprocessor import TRADEPreprocessor    
from inference import inference
from evaluation import _evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/results")
    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5) # maybe use 1e-4
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='adam_w')
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--dev_split_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2) # default was 0.1
    parser.add_argument("--proj_dim", type=int,
                        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()
    
    #args.data_dir = os.environ['SM_CHANNEL_TRAIN']
    #args.model_dir = os.environ['SM_MODEL_DIR']

    # random seed 고정
    set_seed(args.random_seed)


    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file, args.dev_split_ratio)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Define Preprocessor
    if args.model_name_or_path == 'HanBert-54kN-torch':
        tokenizer = HanBertTokenizer.from_pretrained(args.model_name_or_path)
        print('Hanbert tokenizer loaded.')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        print('Tokenizer loaded.')
    processor = TRADEPreprocessor(slot_meta, tokenizer)
    args.vocab_size = len(tokenizer)
    args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.n_gate = len(processor.gating2id)
    config.hidden_size = args.hidden_size
    config.proj_dim = None
    model = TRADE(config, tokenized_slot_meta, slot_meta)
    if args.load_model_dir is not None:
        ckpt = torch.load(args.load_model_dir, map_location="cpu")
        model.load_state_dict(ckpt)
        print("Weight loaded")
    model.to(device)
    print("Model initialized")

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
    # t_total = len(train_loader) * n_epochs 
    t_total = len(train_loader) * 45
    warmup_steps = int(t_total * args.warmup_ratio)
    if args.optimizer == 'adam_w':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        print("Using AdamW optimizer.")
    elif args.optimizer == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        print("Using MADGRAD optimizer")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

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

    wandb.init(config=vars(args), project="p3-dst")

    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        batch_loss = []
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
            batch_loss.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )
                wandb.log({'batch_loss': loss.item(),})
        
        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)

        for k, v in eval_result.items():
            print(f"{k}: {v}")
            wandb.log({k: v})

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update best checkpoint")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

            torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")