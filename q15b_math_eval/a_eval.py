# cmd: python run.py examples/eval_chat_demo.py -w outputs/demo –debug
# vllm cmd: VLLM_WORKER_MULTIPROC_METHOD=spawn python run.py examples/eval_chat_demo.py -a vllm --debug

# srun -p gpua100 --gres=gpu:1 -t 01:00:00 --pty bash
from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess, math_postprocess_v2
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.models import VLLM

max_token = 2048

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN',
                     prompt='''
You are an expert math assistant. Solve mathematical problems with precision. Provide a concise, step-by-step explanation for each solution to ensure student understanding. Prioritize accuracy and logical consistency.

Question: {problem}
Put your final answer within \\boxed{}.
                     '''),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]

models = [
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='q25_1.5b',
    #     path='Qwen/Qwen2.5-1.5B-Instruct',
    #     max_out_len=max_token,
    #     model_kwargs=dict(trust_remote_code=True),
    #     batch_size=1,
    #     run_cfg=dict(num_gpus=1),
    # )

    # vllm-version distilled model - test
    dict(
        # type=HuggingFacewithChatTemplate,
        type=VLLM,
        abbr='qwen2.5-15b-instruct_expa',
#        abbr='llama3.2-3B',
        # path='Haoran22/qwen2.5-3B-distill-Math-Alpaca',
        # path='/cephyr/users/haoranl/Alvis/Downloads/qwen25_15b_0503',
        path='/mimer/NOBACKUP/groups/naiss2026-4-815/HaoranLi/model_0509_re/expa1',
        # tokenizer_path='/home/ha5083li/Downloads/llm_related/knowledge_distillation_llm/results_0220_CE_SFT_lora8/qwen2.5-3B-distill-0220-ep1-1',
        max_out_len=max_token,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            max_tokens=max_token, 
        ), 
        model_kwargs=dict(
            gpu_memory_utilization=0.9,  # 控制预分配显存比例，放在这里生效
            max_model_len=max_token,          # 整个上下文（输入+输出）的最大长度
            # tp_size=1,                   # 如果有多张显卡可以设置张量并行
            # enforce_eager=True,          # 如果还是报错，建议加上这行禁用 CUDA Graph
        ),
    )
]

datasets = math_datasets

work_dir = "outputs/week15-math_prompt1/0509_q15b_expa"
# work_dir = "outputs/week6-Experiments/0406_Qwen2.5_3B"
