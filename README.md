# <p align="center">LM-CPPF</p>

<h2 align="center">
  <p><a href="https://2023.aclweb.org/">[ACL 2023]</a> Paraphrasing-Guided Data Augmentation for Contrastive Prompt-based Few-shot Fine-tuning</p>
</h2>

<p align="center">
  <br>
  <a href=""><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href=""><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000"></a>
  <a href=""><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=googlesheets&logoColor=FFBB00"></a>
</p>

## Abstract

In recent years, there has been significant progress in developing pre-trained language models for NLP. However, these models often struggle when fine-tuned on small datasets. To address this issue, researchers have proposed various adaptation approaches. Prompt-based tuning is arguably the most common way, especially for larger models. Previous research shows that adding contrastive learning to prompt-based fine-tuning is effective as it helps the model generate embeddings that are more distinguishable between classes, and it can also be more sample-efficient as the model learns from positive and negative examples simultaneously. One of the most important components of contrastive learning is data augmentation, but unlike computer vision, effective data augmentation for NLP is still challenging. This paper proposes LM-CPPF, Contrastive Paraphrasing-guided Prompt-based Fine-tuning of Language Models, which leverages prompt-based few-shot paraphrasing using large language models, such as OPT-175B, for data augmentation. Our experiments in multiple text classification benchmarks reveal that this augmentation approach is more effective than other methods, including easy data augmentation, back translation, and multiple templates.

## Citation
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{}
```
