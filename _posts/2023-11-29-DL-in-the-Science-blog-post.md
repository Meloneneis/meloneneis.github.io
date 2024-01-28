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
Proteins are essential for the functioning of living organism(provide some good examples). 
They comprise of 20 different types of amino acids linked together
- these amino acids chains are folded leading to a specific shape
- not the amino acid chain but the shapes will give insight on the specific functions of that protein (provide some use cases for predicting protein structures)
- since the discovery around 50 years ago that the structure can be completely inferred by only the amino acid sequence,
scientists worked hard to predict the this structure from that sequence (techniques: magnetic resonance (NMR),5 X-ray crystallography,6 and cryo-electron microscopy (cryo-EM))
- despite their efforts only around 100000 unique proteins structures have been predicted experimentally
- this showcases the complexity of such task, and the grand challenge it proposed
- Critical Assessment of Structure Prediction (CASP), competition since 1994 for every two years to elevate and evaluate progress in protein structure prediction
- show results of casp-14

## AlphaFold2s Architecture
- introduce AlphaFold2 and mention how it (out)performed in CASP and what impact that had
- write about how despite the vastly faster speed predicting protein structures computationally than experimentally, it still is not fast enough to catch up with large scale gene sequencing technology (used to determine protein sequences)
- write about how its function, MSA, Evoformer, Structure Module

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

Self-attention is the key ingredient for the success of multiple 

<figure>
  <img src="/images/mlm.gif" alt="The beautiful MDN logo.">
  <figcaption>
    A training step via masked language modeling for one amino acid sequence (<b>animated</b>) <br>
    Optional templates removed from AlphaFold2 Architecture for simplicity.
  </figcaption>
</figure>

Transformers made it possible to parallelize 
- short introduction to Transformers (before word embeddings, but static, now dynamic due to attention)
- building blocks of transformers (high level explanation of body and head, maybe decoder/encoder too), attention (medium level explanation), masked language modeling as training objective (medium level explanation)
- also explain evaluation technique: perplexity
- example on what attention does in the context of text (highlight how words can have different meanings depending on the context)
- showcase use cases of transformers (new state of the art, used for a lot of diverse NLP tasks, e.g. text generation, language translation, summarization etc.)

## Contact Map for Protein 3LYW

- ![Contact Map for Protein 3LYW](/images/Contact-Map.png) [Question: Using images from the paper allowed?]
- describe map [unclear from the paper which model they used to make the predictions in contact, but I assume they used the biggest model to do the predictions, i.e. 15B params model]

## Intuition on why it does work so well


## What are the next steps?
Understand the impact of Scale
Attaching folding head to ESM-2



## From Millions to Billions Parameters: Understanding the Impact of Scale
- short introduction to scaling and description of following sub-sections
- MENTION that the following experiments are conducted to highlight the capabilities of the models body (i.e. ESM-2, the base model with no (sophisticated) head attached)
### Impact of number of evolutionarily related sequences in training data
- ![Figure 1B](/images/Figure1BC.png) [describe and interpret]
### Correlation between Language Perplexity and contact accuracy
- ![Figure 1D](/images/Figure1D.png) [describe and interpret]
### Performance on Protein Structure Predictions at Atomic Resolution
- ![Figure 1E](/images/Figure1E.png) [describe and interpret]

## ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction
- mention how it is clear now by the prior done analysis that the model body is able to internalize evolutionarily patterns in Protein Folding
- ![ESMFold architecture](/images/ESMFold-architecture.png)
- explain the pipeline (what is the pair representation? what does the folding trunk do exactly?)
- [Structure Module I will also explain only on high level, because authors did not go further into detail what it is. Question: Is that fine or should I include more details?]
<figure>
  <img src="/images/Architecture.gif" alt="The beautiful MDN logo.">
  <figcaption>Comparison of ESMFold and AlphaFold2 Architecture (<b>animated</b>)</figcaption>
</figure>
## Comparison to SOTA Models: AlphaFold-2 (and RosettaFold)
- Explain metric used: TM-Score, LDDT
- shortly describe the datasets used: CASP-14 and CAMEO
- ![Comparison](/images/Comparison) [describe and interpret]
- ![Figure2C](/images/Figure2C) [describe and interpret]
- maybe mention some limitation due to worse performance especially in CASP-14

## Scientific Contribution: ESM Metagenomic Atlas
- explain what metagenomic is and why sequence prediction was not done until this paper (metagenomic proteins not well studied, as this was not first priority when experimentally predicting structures -> no multiple sequence alignment possible)
- prediction of over 617M sequences
- mention the different confidence levels on these sequences
- write about potential advancements possible made by knowing the structure of metagenomic proteins

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
