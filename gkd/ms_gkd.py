#!/usr/bin/env python
# coding: utf-8

import argparse
from swift.pipelines import rlhf_main


def build_swift_rlhf_args(cli_args: argparse.Namespace) -> list[str]:
    args = [
        '--rlhf_type',
        'gkd',
        '--model',
        cli_args.model,
        '--teacher_model',
        cli_args.teacher_model,
        '--lmbda',
        str(cli_args.lmbda),
        '--beta',
        '1.0',  # ��Ӧ KL(Teacher || Student)
        '--seq_kd',
        'false',  # ʹ�����ݼ��е� response �������� (Off-policy)
        '--tuner_type',
        'lora',
        '--dataset',
        cli_args.train_dataset,
        '--val_dataset',
        cli_args.val_dataset,
        '--torch_dtype',
        'bfloat16',
        '--num_train_epochs',
        str(cli_args.num_train_epochs),
        '--per_device_train_batch_size',
        str(cli_args.per_device_train_batch_size),
        '--per_device_eval_batch_size',
        str(cli_args.per_device_eval_batch_size),
        '--learning_rate',
        str(cli_args.learning_rate),
        '--lora_rank',
        str(cli_args.lora_rank),
        '--lora_alpha',
        str(cli_args.lora_alpha),
        '--lora_dropout',
        str(cli_args.lora_dropout),
        '--target_modules',
        'all-linear',
        '--gradient_accumulation_steps',
        str(cli_args.gradient_accumulation_steps),
        '--eval_steps',
        str(cli_args.eval_steps),
        '--save_steps',
        str(cli_args.save_steps),
        '--save_total_limit',
        str(cli_args.save_total_limit),
        '--logging_steps',
        str(cli_args.logging_steps),
        '--max_length',
        str(cli_args.max_length),
        '--output_dir',
        cli_args.output_dir,
        '--warmup_ratio',
        str(cli_args.warmup_ratio),
        '--dataloader_num_workers',
        str(cli_args.dataloader_num_workers),
        '--dataset_num_proc',
        str(cli_args.dataset_num_proc),
        '--load_from_cache_file',
        'true',
        '--deepspeed',
        cli_args.deepspeed,
        '--attn_impl',
        cli_args.attn_impl,
        '--save_only_model',
        'true',
        '--split_dataset_ratio',
        '0',
        '--report_to',
        'none',
        '--use_hf',
        'true' if cli_args.use_hf else 'false',
    ]
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Qwen2.5 GKD distillation with ms-swift.')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--teacher_model', type=str, default='Qwen/Qwen2.5-Math-7B-Instruct')
    parser.add_argument('--lmbda', type=float, default=0.5, help='Weight of distillation loss')
    parser.add_argument('--train_dataset', type=str, default='data/Distilled_Data_Qwen14B/0401_MATH_1k.json')
    parser.add_argument('--val_dataset', type=str, default='data/MATH_val_1k.json')
    parser.add_argument('--output_dir', type=str, default='output/qwen25_gkd_multinode')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=20)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    parser.add_argument('--dataset_num_proc', type=int, default=8)
    parser.add_argument('--deepspeed', type=str, default='zero2', choices=['zero2', 'zero3'])
    parser.add_argument('--attn_impl', type=str, default='sdpa')
    parser.add_argument('--use_hf', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    swift_args = build_swift_rlhf_args(cli_args)
    rlhf_main(swift_args)




