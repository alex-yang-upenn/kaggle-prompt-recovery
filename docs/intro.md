```{image} ./images/kaggle.png
:target: https://www.kaggle.com
:align: center
:alt: Kaggle
:align: center
```

 # Prompt Recovery Competition

**I'm sharing the Notebooks I used for Kaggle's LLM Prompt Recovery Competition**

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<TODO>) ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg) ![kaggle-badge](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white) ![pytorch-badge](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![keras-badge](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

The goal of this competition is to recover the prompt that an LLM used to rewrite or make stylistic changes to text. The input is the original and the rewritten text, and the output is the prompt. For more details, refer to: [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/overview) 📑 🗂️

## Approaches
There are various ways to approach the prompt recovery task. Two techniques were at the core of the most competitive solutions. 

### Mean Prompt
The first involves finding a "mean" prompt&mdash;a single string of words which, when compared against a diverse array of prompts, yields the highest average similarity score.

```{figure} ./images/beam_search.png
---
name: beam-search
---
A diagram example of a Beam Search, a technique that can be used to find a high scoring mean prompt
```

### LLM
The second, which will be demonstrated in this notebook, involves fine-tuning LLMs to assist in prompt recovery.

```{figure} ./images/lora.png
---
name: lora
---
A diagram demonstrating how LoRA can be used to update weights whilst requiring only a low amount of new parameters
```

### Optimizations

Top leaderboard scores were achieved by combining and adding various optimizations to these approaches. 

For example, with LLM-based approaches, top solutions explored:
- Different models 
    - Mistral/Mixtral, Phi, etc.
- Better datasets
    - As the competition progressed, datasets with large numbers of sample rewrite prompts were collated and published
- Different ways of prompting LLMs 
    - Varying the instruction templates, System prompts, etc.
    - Multi-shot prompting 
- Post-generation modifications
    - Concatenating the responses of several LLM's together
    - Concatenating a "mean" prompt to the LLM response.

Since notebook is meant as an introduction to the core technique of fine-tuning, we will not cover all the possible optimizations. We choose a single Instruction template, Adapter, and base model ([Gemma-7b-it](https://www.kaggle.com/models/google/gemma/transformers/7b-it)). The rewrite prompts were sourced from various publicly available competition datasets.

:::{seealso}
Another high-scoring technique emerged late in the competition, discovered with adversarial attacks against the `sentence-t5` + `sharpened cosine similarity` scoring methodology.

Two key weaknesses were discovered:

1. When comparing similar sentences, adding the `<\s>` token tended to pull the cosine similarity towards 0.9 (corresponding to sharpened cosine similarity of $0.9^{3} \approx 0.73$). For most predictions, this meant an artificial increase in scoring.

2. The Tensorflow `sentence-t5` variant used to score submissions protected against `<\s>` tokens by tokenizing each character separately. However, competitors discovered that `lucrarea` was tokenized in a way that caused similar effects to `<\s>`.

All of the top submissions ended up appending strings containing `lucrarea`. These notebooks will not cover this strategy, as it is a quirk unique to `sentence-t5` and this scoring method. The 1st Place notebook author and pioneer of this strategy discusses it [here](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343).
:::


## Notebook Contents
This will be a three-part guide that produces a complete solution for the Kaggle competition. 

The guide demonstrates how to leverage Quantization and [Parameter Efficient Fine-Tuning](https://www.theaidream.com/post/fine-tuning-large-language-models-llms-using-peft) to train LLMs within Kaggle's free compute resource limitations. Specifically, [LoRA and QLoRA](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) methods are used. The notebooks include introductions to popular libraries for NLP&mdash;Keras, Pytorch, and Hugging Face&mdash;as well as explanations of the underlying techniques and algorithms the libraries help implement. 

:::{tip}
Quantization and LoRA finetuning are incredibly powerful and versatile techniques. They allow developers to create and train custom LLMs, achieving [excellent performance](https://crfm.stanford.edu/2023/03/13/alpaca.html) on specialized tasks, whilst utilizing relatively little compute resources. Although this notebook focuses on a specific Kaggle competition, the techniques covered here are applicable in almost all LLM finetuning projects.
:::

The notebooks are written to be run in Kaggle environments, using the free 2x [T4 GPUs](https://resources.nvidia.com/en-us-gpu-resources/t4-tensor-core-datas?lx=CPwSfP&_gl=1*240zye*_gcl_au*MTY0NjA3MDg5NC4xNzE2ODY1NzIy) and [TPU VM v3-8](https://www.kaggle.com/docs/tpu) accelerators. Links to the versions of these Notebooks that have been ran to completion on Kaggle are given below:
- ### [***Part 1 - Generate Training Data***](https://www.kaggle.com/code/chuhuayang/prompt-recovery-pt-1-generate-training-data) 🗃️
- ### [***Part 2 - Fine-Tuning***](https://www.kaggle.com/code/chuhuayang/prompt-recovery-pt-2-fine-tuning) 🛠️
- ### [***Part 3 - Evaluation***](https://www.kaggle.com/code/chuhuayang/prompt-recovery-pt-3-evaluation) 🧪

The base model, trained Adapter, and training data can all be easily accessed from thes Kaggle links.

## Support
For questions, suggestions, or comments, reach out to me via [email](mailto:chukyang@sas.upenn.edu).

## Credits
Thank you to all members of the Kaggle community, in particular the active contributors in this competition who documented and published their various creative approaches.