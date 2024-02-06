---
title: 'Evolutionary-scale prediction of atomic level protein structure with a language model'
date: 2023-11-20
permalink: /posts/2023/11/DL-in-the-Science/
tags:
  - Deep Learning
  - Protein Structure
  - Bioinformatics
---
# Table of Contents
1. [Introduction](#introduction)
   1. [Understanding Protein Folding](#protein-folding)
   2. [AlphaFold2: 60-years old challenge solved](#alphafold2-60-years-old-challenge-solved)
2. [Limitation of AlphaFold2: The need for Speed and Independence](#limitation-of-alphafold2-the-need-for-speed-and-independence)
3. [Methodology](#methodology)
   1. [ESM-2: Unlocking the Protein Secrets via Transformers](#esm-2-unlocking-the-protein-secrets-via-transformers)
   2. [ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction](#esmfold-attaching-the-head-to-esm-2-for-protein-structure-prediction)
4. [Experiment Results of ESM-2 and ESMFold](#experiment-results-esm-2-and-esmfold)
   1. [ESM-2: Contact Map for Protein 3LYW](#contact-map-for-protein-3lyw)
   2. [ESM-2: Understanding the behind-the-scenes: An intuitive perspective](#understanding-the-behind-the-scenes-an-intuitive-perspective)
   3. [ESM-2: From Millions to Billions Parameters: Understanding the Impact of Scale](#from-millions-to-billions-parameters-understanding-the-impact-of-scale)
   4. [ESMFold: Comparison to SOTA Models: AlphaFold-2 and RosettaFold](#comparison-to-sota-models-alphafold-2-and-rosettafold)
5. [Scientific Contribution: ESM Metagenomic Atlas](#scientific-contribution-esm-metagenomic-atlas)
6. [Conclusion](#conclusion)
   1. [Key Takeaways](#key-takeaways)
   2. [Final Remark](#final-remark)

## Introduction
### Protein Folding
- Protein Definition and Examples
- Tertiary Shape from primary shape
- experimental methods for determining proteins (EM-Cryo...)
- caveats of experimental methods
- CASP
- AlphaFold2
### AlphaFold2: 60-years old challenge solved

If you're a machine learning enthusiast like me, you've probably heard of AlphaFold2. When DeepMind - the
team behind AlphaFold2 - first published their work, it made headlines around the world. And rightly so, as it
provided a solution to a 60-year-old grand challenge in biology, namely protein folding.

So what was all the fuss about, and why is it such a big deal? To understand, we need to take a closer look at the
challenging task of protein folding.
### AlphaFold2 to the rescue
- advantages of AlphaFold2 vs experimental methods
- write about novel architecture (high-level)
- write that AlphaFold2 is not perfect and transition to next section

## Limitation of AlphaFold2: The Need for Speed and Independence
As mentioned above, experimentally predicting the tertiary structure of a protein from its amino acid sequence is an expensive and time-consuming task, but determining the primary structure, i.e. the amino acid sequence itself, isn't. This is demonstrated
by the ever-growing protein sequence databases. In other words, AlphaFold2's inference speed can't keep up with the
growing size of the available primary amino acid sequences. This is mainly due to the time-consuming search for similar proteins during MSA. 
In addition, AlphaFold2 won't be able to handle novel protein sequences as well due to the lack of evolutionarily related sequences in the database.

What if we could simply eliminate the need for multiple sequence alignment altogether, thus eliminating the rather time-consuming
search and dependence on evolutionarily related sequences? What if there was a way to integrate such information in a more
end-to-end fashion? If you are a machine learning expert, or even better an NLP expert, you might know where this is going.
The answer to these questions is another language model that encapsulates all knowledge about amino acids.
You can think of it as a model that learns the language of proteins. Sounds intuitive, doesn't it? ESMFold - the
work by Meta - has done just that, and to be more precise, it has used the state-of-the-art Transformer Encoder
architecture to do so. Since there is already a lot of good content explaining how this architecture works, I will just
give a quick refresher on how and why using a transformer-based architecture is such a good idea here.

## Methodology
### ESM-2: Unlocking the Protein Secrets via Transformers
Transformers were already introduced in 2017 by the paper "Attention is all you need" by Vaswani et al. To date, they remain the state-of-the-art in all sorts of
domains such as natural language, computer vision (vision transformers), and also protein folding. The original
transformers consist of an encoder and a decoder, but it is also possible to use them alone with great success. For protein folding
we will only look at the encoder part of the transformers.

Self-attention is the key ingredient for the success of multiple ...

Masked Language Modeling is the pre-training task...
<figure>
  <img src="/images/mlm.gif" alt="The beautiful MDN logo.">
  <figcaption style="text-align: center;">
    Fig. 1 <b>(animated and own creation)</b>
  </figcaption>
</figure>

## Experiment Results of ESM-2 and ESMFold
### ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction
- mention that the head is adopted version of AlphaFold2 
- structure module same, Folding block is simplified version of Evoformer
- simplified because seq represention is only one dimensional and in AlphaFold2 it is two dimensional (due to MSA), so for AlphaFold2
we need axial attention whereas in ESMFold we only need normal self-attention
- ESM-2 equivalent to MSA in terms of use case
<figure style="text-align: center;">
  <img src="/images/Architecture.gif" alt="The beautiful MDN logo.">
  <figcaption>Fig. 2 <b>(animated and own creation, but Protein Structure from paper)</b></figcaption>
</figure>

### Understanding the behind-the-scenes: an intuitive Perspective
To understand how the contact map can be so well predicted for proteins, we first have to understand what the ESM-2 model replaces, namely MSA.
In MSA, when there are lots of evolutionarily related sequences available, it ...

So coming back to ESM-2 where we don't do MSA. During MLM the model tries to predict masked amino acids. And to decrease the loss, it has to do well on
all the amino acid sequences we are giving it during training, i.e. it has to do well on billions of amino acid sequences. Intuitively, this can only be
achieved if the model learns biological properties of each amino acids. Let me give you an example to clarify what is happening behind the scene.
<div style="display: flex;">
  <figure style="width:25%; margin-right: 10px;">
    <img src="/images/coevolution1.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 3 <b>(animated and own creation)</b></figcaption>
  </figure>
  <figure style="width:25%; text-align: center">
    <img src="/images/coevolution2.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 4 <b>(animated and own creation)</b></figcaption>
  </figure>
</div>

<figure style="width:25%;">
  <img src="/images/mlm_intuition.png" alt="AlphaFold2 Architecture">
  <figcaption style="text-align: center;">Fig. 5 <b>(animated and own creation)</b></figcaption>
</figure>


### From Millions to Billions Parameters: Understanding the Impact of Scale
![Comparison](/images/Scale.jpg) 
[describe and interpret]
![Comparison](/images/Scale2.jpg) 
[describe and interpret]

### Comparison to SOTA Models: AlphaFold-2 and RosettaFold
- Explain metric used: TM-Score
- shortly describe the datasets used: CASP-14 and CAMEO
- maybe mention some limitation due to worse performance especially in CASP-14 (Casp is competition, usually in competitions things are made difficult, i.e. complex protein structure used)
![Comparison](/images/Comparison.jpg) [describe and interpret]

## Scientific Contribution: ESM Metagenomic Atlas
Two advantages can be drawn over AlphaFold2: first, there is the speed-up in protein folding and second, no reliance on evolutionarily related patterns via a genetic database lookup.
Both these advantages were leveraged to create the first metagenomic atlas, at the scale of hundreds of millions of proteins. To put this number into perspective, only 200.000 protein structures were
experimentally predicted, and via AlphaFold2
- explain what metagenomic is and why sequence prediction was not done until this paper (metagenomic proteins not well studied, as this was not first priority when experimentally predicting structures -> no multiple sequence alignment possible)
- prediction of over 617M sequences
- mention the different confidence levels on these sequences
- write about potential advancements possible made by knowing the structure of metagenomic proteins
- link of official blogpost for ESM Metagenomic Atlas: <a href="https://ai.meta.com/blog/protein-folding-esmfold-metagenomics">https://ai.meta.com/blog/protein-folding-esmfold-metagenomics</a>

## Conclusion
### Key Takeaways
AlphaFold2 outperforms ESMFold across all datasets. However, the reliance on MSA for AlphaFold2 hampers its performance for
 novel protein sequences. This was particularly evident in the ablation study, where no evolutionarily related protein
sequences were considered, i.e. mimicking a novel protein sequence. In addition, ESMFold's folding speed makes it easier to
keep pace with the ever-increasing amount of primary protein sequences available, and is therefore able to create the first metagenomic database.

Using a LLM for protein knowledge proves to be a viable alternative to genetic database searches in MSA. However, training this transformer-encoder model through MLM appears to miss out on some evolutionarily relevant information.
Intuitively, this discrepancy makes sense, as MSA explicitly provides valuable insights into co-evolution of amino acid pairs and evolutionarily conserved amino acids in an explicit manner.
In contrast, ESMFold relies on ESM-2 to convey this knowledge implicitly. While MLM should easily infer information about amino acid charges, a fundamental physical property in all sequences, it falls short in capturing all knowledge inferred by MSA.
However, considering the simplicity of the pre-training task, ESMFold did perform better than one would expect.


### Final Remark
Unfortunately, we won't be seeing any further research on ESMFold, as the <a href="https://aibusiness.com/nlp/meta-lays-off-team-behind-its-protein-folding-model">team behind it has been completely let go from Meta</a>.
Meta wants to focus on products that can be commercialised and ESMFold just wasn't it. I think this is
rather unfortunate scenario, as I suspect that ESMFold2 was never meant to be the end of it. My reasoning stems from the
simple pre-training objective and the standard BERT architecture. For instance, further research could have been done to create a 
more sophisticated pre-training objective or architecture that incorporates biological properties better than standard MLM 
similar to the novel geometric attention update incorporated in Evoformer by DeepMind.

------
