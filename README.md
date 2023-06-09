# <p align="center">LM-CPPF</p>

<h2 align="center">
  <p><a href="https://2023.aclweb.org/">[ACL 2023]</a> Paraphrasing-Guided Data Augmentation for Contrastive Prompt-based Few-shot Fine-tuning</p>
</h2>

<p align="center">
  <br>
  <a href="https://arxiv.org/pdf/2305.18169.pdf"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="https://youtu.be/VNj-eGR6-x8"><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000"></a>
  <a href="https://docs.google.com/presentation/d/1-lU-YrwveNzL_2oRHHqAlQT8CycmUWLomCs1YcOO_6c/edit?usp=sharing"><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=googlesheets&logoColor=FFBB00"></a>
</p>

## Intro
This repo covers the implementation of the following paper:  **[LM-CPPF: Paraphrasing-Guided Data Augmentation for Contrastive Prompt-Based Few-Shot Fine-Tuning](https://arxiv.org/pdf/2305.18169.pdf)** by [Amirhossein Abaskohi](https://amirabaskohi.github.io/), [Sascha Rothe](https://research.google/people/107480/) and [Yadollah Yaghoobzadeh](https://yyaghoobzadeh.github.io/), accepted to ACL 2023.

![image](https://github.com/AmirAbaskohi/LM-CPPF/assets/50926437/014681fc-d267-42a4-bbdc-1f199327d1e4)

Our code is  heavily borrowed from [LM-BFF](https://github.com/princeton-nlp/LM-BFF) and [SupCon](https://github.com/HobbitLong/SupContrast) (```/src/losses.py```).

## Abstract

In recent years, there has been significant progress in developing pre-trained language models for NLP. However, these models often struggle when fine-tuned on small datasets. To address this issue, researchers have proposed various adaptation approaches. Prompt-based tuning is arguably the most common way, especially for larger models. Previous research shows that adding contrastive learning to prompt-based fine-tuning is effective as it helps the model generate embeddings that are more distinguishable between classes, and it can also be more sample-efficient as the model learns from positive and negative examples simultaneously. One of the most important components of contrastive learning is data augmentation, but unlike computer vision, effective data augmentation for NLP is still challenging. This paper proposes LM-CPPF, Contrastive Paraphrasing-guided Prompt-based Fine-tuning of Language Models, which leverages prompt-based few-shot paraphrasing using large language models, such as OPT-175B, for data augmentation. Our experiments in multiple text classification benchmarks reveal that this augmentation approach is more effective than other methods, including easy data augmentation, back translation, and multiple templates.

## Requirements

This repo was tested with Ubuntu 18.04.5 LTS, Python 3.7, PyTorch 1.6.0, and CUDA 10.1. You will need a 24 GB GPU for experiments with RoBERTa-base. We run our experiments on Nvidia RTX-3090, but Nvidia A100 also works.

## Download data
We use pre-processed datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP) from  [LM-BFF](https://github.com/princeton-nlp/LM-BFF). LM-BFF offers helpful scripts for downloading and preparing the dataset. Simply run the commands below.
```shell
cd data
bash download_dataset.sh
```
Then use the following command to generate 16-shot datasets we used in the study.
```shell
python tools/generate_k_shot_data.py
```

## Paraphrasing prompt-generation

### Prompt Generation
Navigate to the `few_shot_paraphrasing_data_preparation` directory.

Run the prompt_generator.py file to generate prompts required for the language models. Use the following command:

```shell
python prompt_generator.py [demonstration_template_number] [task_name] [OPTIONAL:instruction_template_number]
```
Replace `[demonstration_template_number]` with the desired template number, `[task_name]` with the task name, and `[OPTIONAL:instruction_template_number]` with the optional instruction template number. The last parameter is optional.

### Paraphrasing

After generating the prompts, you can run the specific code for paraphrasing by selecting one of the available files listed below:

* `paraphrase_with_gpt3.py`
* `paraphrase_with_gpt2_ft.py`
* `paraphrase_with_gpt2.py`


<b>Note</b>: Please be aware that the usage of the ALPA meta OPT-175B model, which was available during the preparation of this paper, is no longer supported. Our experiments were conducted using the ALPA system with the bot we created. Kindly note that the current code does not include access to the ALPA meta OPT-175B model.

To run the code use the following command:

```shell
python paraphrase_file.py [paraphrasing_file] [task_name]
```

Replace `paraphrasing_file.py` with the name of one of the three files mentioned above, and `[task_name]` with the desired task name.

Once the paraphrase file is executed, you can proceed with prompt-based fine-tuning which the instruction are in the next part.

## Running our fine-tuning
The primary prompts (templates) used for tasks have been pre-defined in ```run_experiments.sh```. The auxiliary templates used when generating multi-views of inputs for contrastive learning can be found in ```/auto_template/$TASK```.

Assuming you have one GPU in you system, we show an example of running our fine-tuning on SST-5 (random templates and random demonstrations for "augmented views" of inputs).

```shell
for seed in 13 21 42 87 100   #### random seeds for different train-test splits
do
    for bs in 40   #### batch size
    do
        for lr in 1e-5    #### learning rate for MLM loss
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=exp \
                TYPE=prompt-demo \
                TASK=sst-5 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/
```
Our framework also applies to prompt-based method without demonstrations, i.e., ```TYPE=prompt``` (In this case, we only randomly sample templates for generating "augmented views"). The results are saved in ```log```.



Using RoBERTa-large as the base model requires 4 GPUs, each with 48 GB of memory. You need to first edit Line 20 in ```src/models.py``` to be ```def __init__(self, hidden_size=1024)```.

```shell
for seed in 13 21 42 87 100   #### random seeds for different train-test splits
do
    for bs in 10   #### batch size for each GPU, total batch size is then 40
    do
        for lr in 1e-5    #### learning rate for MLM loss
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=exp \
                TYPE=prompt-demo \
                TASK=sst-5 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/
```



## Collecting results
```
python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-5', 'few_shot_type': 'prompt-demo'}"
```
It will collect the results from ```log``` and compute the mean and standard deviation over those 5 train-test splits.

## Contacts
For any questions, please contact authors.


## Acknowlegements
Thanks to [LM-BFF](https://github.com/princeton-nlp/LM-BFF) and [SupCon](https://github.com/HobbitLong/SupContrast), for the preliminary implementations.

## Citation
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{abaskohi-etal-2023-lm,
    title = "{LM}-{CPPF}: Paraphrasing-Guided Data Augmentation for Contrastive Prompt-Based Few-Shot Fine-Tuning",
    author = "Abaskohi, Amirhossein  and
      Rothe, Sascha  and
      Yaghoobzadeh, Yadollah",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.59",
    pages = "670--681",
    abstract = "In recent years, there has been significant progress in developing pre-trained language models for NLP. However, these models often struggle when fine-tuned on small datasets. To address this issue, researchers have proposed various adaptation approaches. Prompt-based tuning is arguably the most common way, especially for larger models. Previous research shows that adding contrastive learning to prompt-based fine-tuning is effective as it helps the model generate embeddings that are more distinguishable between classes, and it can also be more sample-efficient as the model learns from positive and negative examples simultaneously. One of the most important components of contrastive learning is data augmentation, but unlike computer vision, effective data augmentation for NLP is still challenging. This paper proposes LM-CPPF, Contrastive Paraphrasing-guided Prompt-based Fine-tuning of Language Models, which leverages prompt-based few-shot paraphrasing using generative language models, especially large language models such as GPT-3 and OPT-175B, for data augmentation. Our experiments on multiple text classification benchmarks show that this augmentation method outperforms other methods, such as easy data augmentation, back translation, and multiple templates.",
}

```
