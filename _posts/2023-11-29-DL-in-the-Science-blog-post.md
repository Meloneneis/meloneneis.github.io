---
title: 'Evolutionary-scale prediction of atomic level protein structure with a language model'
date: 2023-11-20
permalink: /posts/2023/11/DL-in-the-Science/
tags:
  - Deep Learning
  - Protein Structure
  - Bioinformatics
---
# Learning Amino Acid Properties via a Language Model for Protein Folding

## AlphaFold2: 60-years old challenge solved

If you're a machine learning enthusiast like me, you've probably heard of AlphaFold2. When DeepMind - the
team behind AlphaFold2 - first published their work, it made headlines around the world. And rightly so, as it
provided a solution to a 60-year-old grand challenge in biology, namely protein folding.

So what was all the fuss about, and why is it such a big deal? To understand, we need to take a closer look at the
challenging task of protein folding.

## Protein Folding
- Protein Definition and Examples
- Tertiary Shape from primary shape
- experimental methods for determining proteins (EM-Cryo...)
- caveats of experimental methods
- CASP
- AlphaFold2

## AlphaFold2 to the rescue
- advantages of AlphaFold2 vs experimental methods
- write about novel architecture (high-level)
- write that AlphaFold2 is not perfect and transition to next section

## The Need for Speed and Independence
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

## Bert-style Transformer for ESM-2
Transformers were already introduced in 2017 by the paper "Attention is all you need" by Vaswani et al. To date, they remain the state-of-the-art in all sorts of
domains such as natural language, computer vision (vision transformers), and also protein folding. The original
transformers consist of an encoder and a decoder, but it is also possible to use them alone with great success. For protein folding
we will only look at the encoder part of the transformers.

Self-attention is the key ingredient for the success of multiple ...

Masked Language Modeling is the pre-training task...
<figure>
  <img src="/images/mlm.gif" alt="The beautiful MDN logo.">
  <figcaption>
    A training step via masked language modeling for one amino acid sequence (<b>animated</b>)
  </figcaption>
</figure>

## Contact Map for Protein 3LYW



## Intuition on why it does work so well
To understand how the contact map can be so well predicted for proteins, we first have to understand what the ESM-2 model replaces, namely MSA.
In MSA, when there are lots of evolutionarily related sequences available, it ...

So coming back to ESM-2 where we don't do MSA. During MLM the model tries to predict masked amino acids. And to decrease the loss, it has to do well on
all the amino acid sequences we are giving it during training, i.e. it has to do well on billions of amino acid sequences. Intuitively, this can only be
achieved if the model learns biological properties of each amino acids. Let me give you an example to clarify what is happening behind the scene.

- Give an example: the role of charges in amino acids during masked language modeling

## From Millions to Billions Parameters: Understanding the Impact of Scale
![Comparison](/images/Scale.jpg) 
[describe and interpret]
![Comparison](/images/Scale2.jpg) 
[describe and interpret]

## ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction
- mention that the head is adopted version of AlphaFold2 
- structure module same, Folding block is simplified version of Evoformer
- simplified because seq represention is only one dimensional and in AlphaFold2 it is two dimensional (due to MSA), so for AlphaFold2
we need axial attention whereas in ESMFold we only need normal self-attention
- ESM-2 equivalent to MSA in terms of use case
<figure>
  <img src="/images/Architecture.gif" alt="The beautiful MDN logo.">
  <figcaption>Comparison of ESMFold and AlphaFold2 Architecture (<b>animated</b>)<br>Optional templates removed from AlphaFold2 Architecture for simplicity.</figcaption>
</figure>

## Comparison to SOTA Models: AlphaFold-2 (and RosettaFold)
- Explain metric used: TM-Score, LDDT
- shortly describe the datasets used: CASP-14 and CAMEO
- ![Comparison](/images/Comparison.jpg) [describe and interpret]
- maybe mention some limitation due to worse performance especially in CASP-14

## Scientific Contribution: ESM Metagenomic Atlas
- explain what metagenomic is and why sequence prediction was not done until this paper (metagenomic proteins not well studied, as this was not first priority when experimentally predicting structures -> no multiple sequence alignment possible)
- prediction of over 617M sequences
- mention the different confidence levels on these sequences
- write about potential advancements possible made by knowing the structure of metagenomic proteins
- link of official blogpost for ESM Metagenomic Atlas: https://ai.meta.com/blog/protein-folding-esmfold-metagenomics/

## Conclusion
AlphaFold2 outperforms ESMFold across all datasets. However, the reliance on MSA for AlphaFold2 hampers its performance for
for novel protein sequences. This was particularly evident in the ablation study, where no evolutionarily related protein
sequences were considered, i.e. mimicking a novel protein sequence. In addition, ESMFold's folding speed makes it easier to
"to keep pace with the ever-increasing amount of primary protein sequences available, and is therefore able to create the first metagenomic database.

Overall, the use of an LLM for protein knowledge is a valid alternative to genetic database searches for MSA.
However, training such a transformer-encoder model via MLM doesn't seem to extract all the evolutionarily relevant information available,
but given the highly complex protein folding task, such a simple pre-training objective performs better than one would expect.


## Final Remark
Unfortunately, we won't be seeing any further research on ESMFold, as the <a href="https://aibusiness.com/nlp/meta-lays-off-team-behind-its-protein-folding-model">team behind it has been completely let go from Meta</a>.
Meta wants to focus on products that can be commercialised and ESMFold just wasn't it. I think this is
rather unfortunate scenario, as I suspect that ESMFold2 was never meant to be the end of it. My reasoning stems from the
simple pre-training objective and the standard BERT architecture. For instance, further research could have been done to create a 
more sophisticated pre-training objective that incorporates biological properties better than standard MLM.

------
