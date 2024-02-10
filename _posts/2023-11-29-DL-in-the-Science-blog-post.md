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
   1. [Understanding Protein Folding](#underanding-protein-folding)
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
### Understanding Protein Folding
In this blog post, we'll explore the fascinating realm of protein folding-a process that is fundamental to understanding the intricate functions of these molecular marvels. Proteins, the workhorses of life, perform a range of vital tasks, from catalyzing chemical reactions to transporting molecules within cells. Consider helicase, a protein critical for unwinding DNA during replication, as an example of the diverse roles proteins play in cellular processes.

Proteins are made up of chains of amino acids that are intricately folded into complex three-dimensional structures. This folding process is critical because the shape of a protein largely determines its function. Take helicase, for example, with its distinctive ring-shaped structure through which DNA strands pass, facilitating their unwinding.

Remarkably, even with mutations that alter the sequence of amino acids, proteins can retain their function if their overall shape remains unchanged. This underscores the importance of understanding protein folding and drives myriad efforts to experimentally predict protein structures.

Experimental methods such as X-ray crystallography and nuclear magnetic resonance (NMR) spectroscopy provide insight into protein structures, but with limitations such as cost and time. Despite significant progress, only a fraction of proteins have been experimentally characterized, highlighting the need for initiatives such as the Critical Assessment of Structure Prediction (CASP) to advance research in this area.

In the 14th CASP competition, there one was team that outperformed their competitors by far. This team has created a model for which the grand challenge of cheap and fast protein folding could be considered solved. This team is none other than Google DeepMind with their model AlphaFold2.

### AlphaFold2: 60-years old challenge solved
If you're a machine learning enthusiast like me, than you've probably already heard of AlphaFold2. This model marks the first time machine learning has been successfully used for protein folding.
Here is the thing though, trying to solve the task with a machine learning model has been tried before, but due to the complexity structure folding, it was without much success.
Scientists have tried to use machine learning to ... but the thing about this methodology is that it unstable due to

So how can we make this more stable? DeepMind came up with a plan to do this more stable. Multiple novel things came into the creation of their architecture. Name them.
Later in this blog post, the architecture will of AlphaFold2 will be shown.

With all that said, is AlphaFold2 the end of research? Is the protein folding considered solved now? I would so no, as AlphaFold2, admittedly a very good creation, is not perfect.
This can be contributed to the main two things: 1. its speed in prediction and 2. its dependence on multiple sequence alignment (MSA).

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
Transformers, the new SOTA architecture introduced in the paper "Attention is all you need" by Vaswani et al., have revolutionized various domains, including natural language processing, computer vision with vision transformers, and the field of protein folding. At the heart of transformers' success lie two main components: self-attention and parallelization. 

Self-attention, in essence, allows the model to weigh the importance of different parts of the input sequence in relation to each other, effectively understanding the context within the sequence as a whole. This mechanism ensures that the model can focus on the most relevant parts of the input to make predictions, making it particularly adept at capturing relationships between tokens.

When we discuss transformers in the context of protein folding, we specifically refer to the encoder part of the architecture. This design choice means that we are primarily focused on enriching an input sequence with contextual information, rather than generating new sequences. The encoder achieves this through a training objective known as masked language modeling (MLM).

MLM is a training strategy where random parts of the input sequence are masked from the model during training, and then let the model predict these hidden parts based only on the context provided by the remaining sequence. Figure 1 shows such a training step on a single amino acid chain.

<figure>
  <img src="/images/mlm.gif" alt="The beautiful MDN logo.">
  <figcaption style="text-align: center;">
    Fig. 1 <b>(animated and own creation)</b>
  </figcaption>
</figure>

Needless to say, they weren't just training on a single amino acid sequence, but on ~65 million unique sequences, and thereby parallelization is crucial for training such a model. Intuitively, for the model to perform well on MLM on millions of sequences, it needs to learn biological amino acid properties relevant to the folding mechanism.

To analyse how suitable training such a model for protein knowledge, which they call ESM-2, is two main experiments have been conducted. 

[First](#), the authors mapped the attention scores for every single amino acid pair over the weighted attention map and compared this with the experimentally determined ground truth on the 3LYW protein.

[Secondly](#from-millions-to-billions-parameters-understanding-the-impact-of-scale), the authors want to understand what happens when we scale up the ESM-2 and if this is a valid approach to improve the quality of said knowledge. Specifically, the smaller ESM-2 is compared to the next largest ESM-2 on a set of validation proteins, and there they check the change in perplexity and precision of long-range contact. Perplexity is the number of amino acids the model chooses for the masked amino acid, and precision of long range contact is the number of contact predictions the model got right out of all contact predictions made.


### ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction
For the prediction of tertiary protein structure, the authors used ESM-2 and attached the ESMFold head to it. It consists of a stack folding blocks and then a stack of structure modules. Before the protein structure is finalized, it gets recycled through the whole head multiple times.

If you are familiar with AlphaFold2 than this kind of architecture should sound familiar, because this is just an adapted version of AlphaFold2! Meta did not change the underlying architecture, but simplification were possible because of the use of ESM-2 instead of MSA. In ESM-2 the output is the enriched amino acid sequence representation, which in theory should incorporate evolutionarily related knowledge extracted during MLM, whereas in AlphaFold2 this step still has to be done! Thus in AlphaFold we pass in the sequence representation we are interested in and evolutionarily related sequence representation found via the genetic database search. Figure 2 shows the differences of both ESMFold and Alphafold2 (the optional template lookup is omitted in the figure for simplicity).

<figure>
  <img src="/images/Architecture.gif" alt="The beautiful MDN logo.">
  <figcaption style="text-align: center;">Fig. 2 <b>(animated and own creation, except Protein Structure from paper)</b></figcaption>
</figure>

One key advantage of ESMFold is that as the sequence representation is only one dimensional whereas in AlphaFold2 it is two dimensional. This leads to a simplification of the Evoformer as no axial attention mechanism are needed and only simple self-attention can be applied (hence the changed name to Folding Block). 

Both these models are then the datasets CAMEO and CASP14 on their TM-Score, which measures the similarity between the predicted structure and the ground truth.


## Experiment Results of ESM-2 and ESMFold
### Unsupervised Contact Map for LY3W
The following figure shows the contact map for every amino acid pair in the protein LY3W. The ground truth is depicted in the upper left whereas the contact prediction for the ESM-2 via the weighted attention map is shown in the bottom right half.
<figure>
 <img src="/images/contactmap.jpg" alt="Contact Map for LY3W">
 <figcaption style="text-align: center;">Fig. 3 <b>(Source: Paper)</b></figcaption>
</figure>

Immediately noticeable is that both contact maps are quite similar. This is quite remarkable because the model was not trained on that contact task, hence it is unsupervised! This showcases how the attention patterns in a transformer architecture correlates to the ground truth contact map!
### Understanding the behind-the-scenes: an intuitive Perspective
To grasp why attention scores align closely with the contact map, revisiting the role of multiple sequence alignment (MSA) in AlphaFold2 is beneficial. Two principal properties are extracted from MSAs: evolutionary conservation and co-evolution, both critical for understanding protein structure. Evolutionary conservation is observed when an amino acid remains unchanged across species over evolution, highlighting its significance to the protein's function. Co-evolution occurs when a mutation in one amino acid necessitates a compensatory mutation in another to preserve the protein's structure. These properties impose vital constraints on the protein's ultimate 3D structure, aiding in precise structure prediction (see figure 3 and 4).
<div style="display: flex;">
  <figure style="width:35%; margin-right: 10px;">
    <img src="/images/coevolution1.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 3 <b>(animated and own creation)</b></figcaption>
  </figure>
  <figure style="width:35%; text-align: center">
    <img src="/images/coevolution2.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 4 <b>(animated and own creation)</b></figcaption>
  </figure>
</div>

In masked language models (MLMs), directly extracting these properties is more challenging since they do not process inputs in the same integrated manner as AlphaFold2. Nevertheless, for MLMs to excel with millions of protein sequences, they must implicitly learn some of these properties, albeit not as explicitly as AlphaFold2. Consider a scenario where an amino acid, engaged in a long-range contact with another, is masked. This situation implies the amino acids must have opposite charges; if one is positively charged, the masked one must be negatively charged to facilitate contact. This requirement narrows down the selection of possible amino acids, thus reducing prediction complexity (perplexity).

<figure style="width:35%;">
  <img src="/images/mlm_intuition.png" alt="AlphaFold2 Architecture">
  <figcaption style="text-align: center;">Fig. 5 <b>(animated and own creation)</b></figcaption>
</figure>


### From Millions to Billions Parameters: Understanding the Impact of Scale
<figure style="width:35%;">
  <img src="/images/Scale.jpg" alt="Scale Figure 1">
  <figcaption style="text-align: center;">Fig. 5 <b>(Source: from paper)</b></figcaption>
</figure>
From left to right shows the model starting from 8M parameters to 15B parameters. On the x-axis the smaller model is depicted and on the y-axis the next bigger one. 
Visible from the plots, there is a trend of improving perplexity for almost all proteins shown by the blueish color. Simultaneously, the unsupervised contact precision improves as well for most of the proteins with scale. This showcases that the perplexity is indeed highly correlated with the contact precision. 

Interestingly, in the third and forth plot there are proteins, albeit just a few of them, were the performance decreases a lot. This was not further investigated by the authors, but I can give you a possible high-level explanation to that. 

When a model is scaled up, it learns more complex information, improving performance on most data points in a dataset. However, some data points, like certain proteins in this context, may drastically underperform with larger models. This can happen because these outliers differ significantly from the majority of data. Smaller models might do better with such outliers because they focus on basic, common characteristics found across the dataset. Imagine this scenario with the English and German languages, where most data is in German. At a basic level, both languages follow the simple sentence structure of Subject + Verb + Object. For example, in English, we say "I read books," and in German, "Ich lese Bücher."

This foundational structure is easily captured by smaller models. However, when sentences become more complex, such as those including a modal verb, the structure in German shifts to Subject + (modal verb) + Object + Verb, as in "Ich muss Bücher lesen." However, in English, the sentence structure doesn't change, even with the addition of a modal verb, as seen in "I must read books." This means that smaller models, which focus on basic structures, continue to perform well with these slightly more complex English sentences. If a model is trained mainly on German data and then scaled up, it learns to adjust for the modal verb placement, which negatively affect the performance on the rare English sentences with modal verbs. The scaled-up model is now better at handling complex German sentence structures, i.e. for sentences with modal verbs change structure from: "Subject + Verb + Object" to "Subject + (Modal Verb) + Object + Verb". However, applying this knowledge to the rare English sentences with modal verbs would in turn lead to a performance loss as the structure still is the same even with modal verbs for English.

Coming back to the proteins that perform worse, we can't say for sure why it happens, but what should be clear is that these proteins should differ a lot from the general data and I think it is important to analyse such cases further to make sure that the model always gives a good prediction no matter which protein we want to infer.
<figure style="width:35%;">
  <img src="/images/Scale2.jpg" alt="Scale Figure 2">
  <figcaption style="text-align: center;">Fig. 5 <b>(animated and own creation)</b></figcaption>
</figure>



### Comparison to SOTA Models: AlphaFold-2 and RosettaFold
- Explain metric used: TM-Score
- shortly describe the datasets used: CASP-14 and CAMEO
- maybe mention some limitation due to worse performance especially in CASP-14 (Casp is competition, usually in competitions things are made difficult, i.e. complex protein structure used)
![Comparison](/images/Comparison.jpg) [describe and interpret]

## Scientific Contribution: ESM Metagenomic Atlas
There are two key advantages over AlphaFold2: (i) there is the acceleration of protein folding and (ii) there is no reliance on evolutionary related patterns via a genetic database search.
These two advantages have been exploited to create the first metagenomic atlas on the scale of hundreds of millions of proteins. To put this number into perspective, only 200,000 protein structures have been
were predicted experimentally, and through AlphaFold2 there are currently 200 million protein structures in the AlphaFold database, predicted over a period of 3 years. However,
via ESMFold, the ESM Metagenomic Atlas contains over 600 million protein structures predicted in just <b>2 weeks</b>!

This is the first time that metagenomic proteins from microbes found in soil, seawater or the ocean have been folded. Understandibly so, because
such a feast was not possible before as metagenomic proteins, i.e. from nature, do not have evolutionarily related sequences. This is where ESMFold can shine, 
as it does not rely on such genetic database search for a good protein structure prediction. As a result, more than X number of proteins folded from the metagnomic
atlas are of high-confidence! 

This Metagenomic Atlas unveils new protein structure that can potentially be leveraged for all sorts of tasks (name some).

If you want to find more about ESM Metagenomic Atlas, I can recommend you to read through the official blog post of Meta themselves: <a href="https://ai.meta.com/blog/protein-folding-esmfold-metagenomics">https://ai.meta.com/blog/protein-folding-esmfold-metagenomics</a>

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
