# Language Model Evaluation Harness

## Notice to Users
(as of 6/15/23)
We have a revamp of the Evaluation Harness library internals staged on the [big-refactor](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) branch! It is far along in progress, but before we start to move the `master` branch of the repository over to this new design with a new version release, we'd like to ensure that it's been tested by outside users and there are no glaring bugs.

We’d like your help to test it out! you can help by:
1. Trying out your current workloads on the big-refactor branch, and seeing if anything breaks or is counterintuitive,
2. Porting tasks supported in the previous version of the harness to the new YAML configuration format. Please check out our [task implementation guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/new_task_guide.md) for more information.

If you choose to port a task not yet completed according to [our checklist](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/tasks/README.md), then you can contribute it by opening a PR containing [Refactor] in the name with: 
- A shell command to run the task in the `master` branch, and what the score is
- A shell command to run the task in your PR branch to `big-refactor`, and what the resulting score is, to show that we achieve equality between the two implementations.

Lastly, we'll no longer be accepting new feature requests beyond those that are already open to the master branch as we carry out this switch to the new version over the next week, though we will be accepting bugfixes to `master` branch and PRs to `big-refactor`. Feel free to reach out in the #lm-thunderdome channel of the EAI discord for more information.


## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.
- Task versioning to ensure reproducibility when tasks are updated.

## Install

For model evaluation, we follow the same method in [open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), which evaluate 4 key benchmarks in the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

- To setup the evaluate env, use the following command
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Basic Usage

```bash
### arc_challenge ###
python /home/vmagent/app/deltatuner/lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args pretrained=/home/vmagent/app/LLM/data/Llama-2-7b-hf,peft=/home/vmagent/app/LLM/data/llama-7b-delta-tune,use_accelerate=True,delta=/home/vmagent/app/LLM/data/llama-7b-delta-tune/best_model_structure.txt \
        --tasks arc_challenge  --num_fewshot 25 \
        --batch_size auto --max_batch_size 32 \
        --output_path /home/vmagent/app/LLM/data/llama-7b-delta-tune/llama2-7b-delta-arc_challenge

### truthful_qa ###
python /home/vmagent/app/deltatuner/lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args pretrained=/home/vmagent/app/LLM/data/Llama-2-7b-hf,peft=/home/vmagent/app/LLM/data/llama-7b-delta-tune,use_accelerate=True,delta=/home/vmagent/app/LLM/data/llama-7b-delta-tune/best_model_structure.txt \
        --tasks truthfulqa_mc  --num_fewshot 0 \
        --batch_size auto --max_batch_size 32 \
        --output_path /home/vmagent/app/LLM/data/llama-7b-delta-tune/llama2-7b-delta-truthqa

### hellaswag ###
python /home/vmagent/app/deltatuner/lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args pretrained=/home/vmagent/app/LLM/data/Llama-2-7b-hf,peft=/home/vmagent/app/LLM/data/llama-7b-delta-tune,use_accelerate=True,delta=/home/vmagent/app/LLM/data/llama-7b-delta-tune/best_model_structure.txt \
        --tasks hellaswag  --num_fewshot 10 \
        --batch_size auto --max_batch_size 32 \
        --output_path /home/vmagent/app/LLM/data/llama-7b-delta-tune/llama2-7b-delta-hellaswag

### mmlu ###
python /home/vmagent/app/deltatuner/lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args pretrained=/home/vmagent/app/LLM/data/Llama-2-7b-hf,peft=/home/vmagent/app/LLM/data/llama-7b-delta-tune,use_accelerate=True,delta=/home/vmagent/app/LLM/data/llama-7b-delta-tune/best_model_structure.txt \
        --tasks hendrycksTest*  --num_fewshot 5 \
        --batch_size auto --max_batch_size 32 \
        --output_path /home/vmagent/app/LLM/data/llama-7b-delta-tune/llama2-7b-delta-mmlu
```