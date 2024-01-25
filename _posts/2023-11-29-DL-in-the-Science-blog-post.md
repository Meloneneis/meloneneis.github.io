---
title: 'Evolutionary-scale prediction of atomic level protein structure with a language model'
date: 2023-11-20
permalink: /posts/2023/11/DL-in-the-Science/
tags:
  - Deep Learning
  - Protein Structure
  - Bioinformatics
---
# Evolutionary-scale Prediction of Atomic Level Protein Structure with a Language Model

## Introduction to Prediction of Protein Structures

Proteins provide many essential function in organisms (provide some good examples). Proteins comprises of 20 different types of amino acids linked together
- these amino acids chains are folded leading to a specific shape
- not the amino acid chain but the shapes will give insight on the specific functions of that protein (provide some use cases for predicting protein structures)
- since the discovery around 50 years ago that the structure can be completely inferred by only the amino acid sequence,
scientists worked hard to predict the this structure from that sequence (techniques: magnetic resonance (NMR),5 X-ray crystallography,6 and cryo-electron microscopy (cryo-EM))
- despite their efforts only around 100000 unique proteins structures have been predicted experimentally
- this showcases the complexity of such task, and the grand challenge it proposed
- Critical Assessment of Structure Prediction (CASP), competition since 1994 for every two years to elevate and evaluate progress in protein structure prediction

## AlphaFold: 50-years long challenge solved
- introduce AlphaFold2 and mention how it (out)performed in CASP (show diagram) and what impact that had
- write about how despite the vastly faster speed predicting protein structures computationally than experimentally, it still is not fast enough to catch up with large scale gene sequencing technology (used to determine protein sequences)

## ESMFold: Need for Speed
- Need for Speed addressed by ESMFold (done by omitting the time consuming task of Multiple Sequence Alignment needed in AlphaFold)
- high level explanation of Multiple Sequence Alignment and why AlphaFold incorperated that in their model
- omitting this time consuming procedure could speed up the prediction of protein Structures
- the question now is: without MSA can Protein Folding Models still compete with SOTA Models in terms of accuracy?
- Meta addressed this question in their ESM-2 paper by scaling up to the biggest protein model ever trained (15B params)
- experiments conducted to understand what their model learns at different scales (8M to 15B params) only trained on amino acid chains
- Transition to transformer architecture: A foundational understanding necessary for evaluating ESMFold’s performance without MSA.

## Transformers: the State-of-the-Art Language Model

- short introduction to Transformers (before word embeddings, but static, now dynamic due to attention)
- building blocks of transformers (high level explanation of body and head, maybe decoder/encoder too), attention (medium level explanation), masked language modeling as training objective (medium level explanation)
- also explain evaluation technique: perplexity
- example on what attention does in the context of text (highlight how words can have different meanings depending on the context)
- showcase use cases of transformers (new state of the art, used for a lot of diverse NLP tasks, e.g. text generation, language translation, summarization etc.)


## From Natural Language to the Language of Proteins: The Role of Transformers

- transformers not limited to natural language
- write about the idea of using transformers for predicting the protein structure (i.e. model learns patterns from amino sequences)
- what about attention in amino sequences? (attention actually learns the residue-residue contact map, i.e. the closer a residue (amino acid in the protein sequence after binding) is to another attention the more it pays "attention" to that)
- highlight the amazingness of the capabilities of transformers (it was NOT trained to do so!)
- transition to contact map for protein 3LYW

### Contact Map for Protein 3LYW

- ![Contact Map for Protein 3LYW](/images/Contact-Map.png) [Question: Using images from the paper allowed?]
- describe map [unclear from the paper which model they used to make the predictions in contact, but I assume they used the biggest model to do the predictions, i.e. 15B params model]

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
- draw conclusions from the comparison

------
