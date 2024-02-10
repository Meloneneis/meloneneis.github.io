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
   1. [Understanding Protein Folding](#understanding-protein-folding)
   2. [AlphaFold2: 60-years old challenge solved](#alphafold2-60-years-old-challenge-solved)
2. [Limitation of AlphaFold2: The need for Speed and Independence](#limitation-of-alphafold2-the-need-for-speed-and-independence)
3. [Methodology](#methodology)
   1. [ESM-2: Unlocking the Protein Secrets via Transformers](#esm-2-unlocking-the-protein-secrets-via-transformers)
   2. [ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction](#esmfold-attaching-the-head-to-esm-2-for-protein-structure-prediction)
4. [Experiment Results of ESM-2 and ESMFold](#experiment-results-of-esm-2-and-esmfold)
   1. [ESM-2: Unsupervised Contact Map for Protein 3LYW](#unsupervised-contact-map-for-ly3w)
   2. [ESM-2: Understanding the behind-the-scenes: An intuitive perspective](#understanding-the-behind-the-scenes-an-intuitive-perspective)
   3. [ESM-2: From Millions to Billions Parameters: Understanding the Impact of Scale](#from-millions-to-billions-parameters-understanding-the-impact-of-scale)
   4. [ESMFold: Comparison to SOTA Models: AlphaFold-2 and RosettaFold](#comparison-to-sota-models-alphafold-2-and-rosettafold)
5. [Scientific Contribution: ESM Metagenomic Atlas](#scientific-contribution-esm-metagenomic-atlas)
6. [Conclusion](#conclusion)
   1. [Key Takeaways](#key-takeaways)
   2. [Final Remark](#final-remark)

## Introduction
### Understanding Protein Folding
In this blog post, we'll explore the concept of protein folding, which is essential for understanding how these molecules function. Proteins are like the workhorses of life, carrying out vital tasks such as speeding up chemical reactions and transporting molecules within cells. For instance, there's a protein called helicase that's crucial for unwinding DNA during replication. It's fascinating to see how proteins have diverse roles in cellular processes.

Proteins are like chains made of building blocks called amino acids. They fold up in complicated ways to form specific shapes in three dimensions. This tertiary protein structure is important because the shape of a protein largely contributes to its function. For example for the protein helicase which is shaped like a ring: one DNA strand from the double helix is passed through the helicase ring which leads to the unwinding.

Surprisingly, even if the amino acid sequence changes due to mutations, proteins can still work if their overall shape stays the same. This shows why it's important to understand how proteins fold, which is why scientists are trying lots of ways to predict protein shapes in experiments.

Scientists use methods like X-ray crystallography and nuclear magnetic resonance (NMR) spectroscopy to learn about protein structures, but these methods can be expensive and time-consuming. Hence, just 200000 were experimentally predicted from billions of known amino acid sequences. To further elevate research into that field, the Critical Assessment of Structure Prediction (CASP) was created to find ways to fold proteins in a more affordable and efficient way.

In the 14th CASP competition, there was a team that performed much better than its competitors. This team developed a model that could be seen as have the grand challenge of protein folding solved. That team is none other than Google DeepMind with their model AlphaFold2.
### AlphaFold2: 60-years old challenge solved
If you are interested in machine learning, AlphaFold2 has probably been on your radar. This groundbreaking model stands out as the first successful application of machine learning to the complex challenge of protein folding. Previous attempts to tackle protein folding with machine learning models have fallen short, largely due to the intricate nature of protein structures. Efforts to apply machine learning to this task have often been met with instability and lackluster results.
But how did DeepMind overcome these obstacles to stability in protein folding predictions? They developed a robust strategy, integrating several innovative elements into their model's architecture, which will be explored in more detail later in this blog post.

With all that said, is AlphaFold2 the end of research? Is the protein folding considered solved now? I would so no, as AlphaFold2, admittedly a very good creation, is not perfect.
This can be contributed to the main two things: 1. its speed in prediction and 2. its dependence on multiple sequence alignment (MSA).

## Limitation of AlphaFold2: The Need for Speed and Independence
As mentioned earlier, predicting the three-dimensional structure of a protein from its sequence of amino acids using experiments is both costly and time-intensive. However, determining the primary structure, which is the sequence of amino acids itself, is comparatively easier and faster. This is evident from the continuously expanding databases of protein sequences. In simpler terms, AlphaFold2's speed in making predictions can't match the rapid growth of available primary amino acid sequences. This is largely because searching for similar proteins during Multiple Sequence Alignment (MSA) takes up a lot of time. Moreover, AlphaFold2 may struggle with novel protein sequences due to the absence of closely related sequences in the database.

What if we could skip the need for multiple sequence alignment altogether, cutting out the time-consuming search and reliance on evolutionarily related sequences? What if we could integrate such information more seamlessly? If you're familiar with machine learning or, even better, natural language processing (NLP), you might see where this is headed. The solution lies in another language model that comprehensively understands amino acids - essentially, a model fluent in the language of proteins. It's quite intuitive, isn't it? ESMFold, developed by Meta, has achieved exactly that, leveraging the cutting-edge Transformer Encoder architecture. Since there's already a lot of information available on how this architecture functions, I'll briefly touch on why using a transformer-based approach is such a smart move in this context.
## Methodology
### ESM-2: Unlocking the Protein Secrets via Transformers
Transformers, the new SOTA architecture introduced in the paper "Attention is all you need" by Vaswani et al., have revolutionized various domains, including natural language processing, computer vision with vision transformers, and the field of protein folding. At the heart of transformers' success lie two main components: self-attention and parallelization. 

Self-attention, in essence, allows the model to weigh the importance of different parts of the input sequence in relation to each other, effectively understanding the context within the sequence as a whole. This mechanism ensures that the model can focus on the most relevant parts of the input to make predictions, making it particularly adept at capturing relationships between tokens.

When we discuss transformers in the context of protein folding, we specifically refer to the encoder part of the architecture. This design choice means that we are primarily focused on enriching an input sequence with contextual information, rather than generating new sequences. The encoder achieves this through a training objective known as masked language modeling (MLM).

MLM is a training strategy where random parts of the input sequence are masked from the model during training, and then let the model predict these hidden parts based only on the context provided by the remaining sequence. Figure 1 shows such a training step on a single amino acid chain.

<figure>
  <img src="/images/mlm.gif" alt="The beautiful MDN logo.">
  <figcaption style="text-align: center;">Fig. 1 <b>(animated and own creation)</b></figcaption>
</figure>

Needless to say, they weren't just training on a single amino acid sequence, but on ~65 million unique sequences, and thereby parallelization is crucial for training such a model. Intuitively, for the model to perform well on MLM on millions of sequences, it needs to learn biological amino acid properties relevant to the folding mechanism.

To analyse how suitable training such a model for protein knowledge, which they call ESM-2, is two main experiments have been conducted. 

[First](#), the authors mapped the attention scores for every single amino acid pair over the weighted attention map and compared this with the experimentally determined ground truth on the 3LYW protein.

[Secondly](#from-millions-to-billions-parameters-understanding-the-impact-of-scale), the authors want to understand what happens when we scale up the ESM-2 and if this is a valid approach to improve the quality of said knowledge. Specifically, the smaller ESM-2 is compared to the next largest ESM-2 on a set of validation proteins, and there they check the change in perplexity and precision of long-range contact. Perplexity is the number of amino acids the model chooses for the masked amino acid, and precision of long range contact is the number of contact predictions the model got right out of all contact predictions made.


### ESMFold: Attaching the head to ESM-2 for Protein Structure Prediction
In their approach to predicting the tertiary structure of proteins, the researchers employed ESM-2, augmenting it with a component known as the ESMFold head. This head is composed of a sequence of folding blocks followed by a series of structure modules. To refine the protein structure, the model iteratively processes it through the ESMFold head multiple times, a technique known as recycling, which helps to progressively improve the accuracy of the structure prediction.

For those acquainted with AlphaFold2's design, the architecture of ESMFold may seem quite familiar, as it is a modified version of AlphaFold2's setup. However, Meta's implementation has been simplified by leveraging ESM-2's capabilities, thereby bypassing the need for multiple sequence alignments (MSA). ESM-2 outputs an amino acid sequence representation that is already enhanced, theoretically embedding evolutionary relationships learned during MLM. In contrast, AlphaFold2 necessitates an additional step to incorporate evolutionarily related sequence information obtained through genetic database searches. Consequently, while AlphaFold2 inputs both the target sequence representation and its evolutionarily related counterparts, ESMFold requires only the former. The distinctions between ESMFold and AlphaFold2's methodologies are depicted in Figure 2, which excludes the optional template lookup for clarity.

<figure>
  <img src="/images/Architecture.gif" alt="The beautiful MDN logo.">
  <figcaption style="text-align: center;">Fig. 2 <b>(animated and own creation, except Protein Structure from paper)</b></figcaption>
</figure>

Figure two illustrates that ESMFold takes two inputs: a pair representation and a sequence representation, in contrast to the multiple sequence alignment (MSA) representation used in AlphaFold's Evoformer. The folding block in ESMFold is a simplified version of the Evoformer, eliminating the need for axial attention thanks to the one-dimensional nature of the sequence representation. The pair representation acts as a detailed blueprint, laying out the potential interactions between amino acid pairs. In the folding block, the pair representation is regularly updated to reflect the latest changes in the sequence data. This process involves a back-and-forth adjustment where updates to the sequence and its geometry are made together, keeping both the sequence information and the spatial relationships between amino acids in sync. After this process, the refined sequence and pair representation is passed to the structure module, which translates the abstract representation of the protein's sequence into concrete three-dimensional coordinates, effectively mapping the protein's structure in space.

Both these models are then compared on datasets CAMEO and CASP14 on their TM-Score, which measures the similarity between the predicted structure and the ground truth.


## Experiment Results of ESM-2 and ESMFold
### Unsupervised Contact Map for LY3W
The following figure shows the contact map for every amino acid pair in the protein LY3W. The ground truth is depicted in the upper left whereas the contact prediction for the ESM-2 via the weighted attention map is shown in the bottom right half.
<figure style="width:35%; margin-right: 10px;">
 <img src="/images/contactmap.jpg" alt="Contact Map for LY3W">
 <figcaption style="text-align: center;">Fig. 3 <b>(Source: Paper)</b></figcaption>
</figure>

Immediately noticeable is that both contact maps are quite similar. This is quite remarkable because the model was not trained on that contact task, hence it is unsupervised! This showcases how the attention patterns in a transformer architecture correlates to the ground truth contact map!
### Understanding the behind-the-scenes: an intuitive Perspective
To grasp why attention scores align closely with the contact map, revisiting the role of multiple sequence alignment (MSA) in AlphaFold2 is beneficial. Two principal properties are extracted from MSAs: evolutionary conservation and co-evolution, both critical for understanding protein structure. Evolutionary conservation is observed when an amino acid remains unchanged across species over evolution, highlighting its significance to the protein's function. Co-evolution occurs when a mutation in one amino acid necessitates a compensatory mutation in another to preserve the protein's structure. These properties impose vital constraints on the protein's ultimate 3D structure, aiding in precise structure prediction (see figure 3 and 4).
<div style="display: flex;">
  <figure style="width:35%; margin-right: 10px;">
    <img src="/images/coevolution1.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 4<b>(animated and own creation)</b></figcaption>
  </figure>
  <figure style="width:30%; text-align: center">
    <img src="/images/coevolution2.gif" alt="AlphaFold2 Architecture">
    <figcaption style="text-align: center;">Fig. 5 <b>(animated and own creation)</b></figcaption>
  </figure>
</div>

In masked language models (MLMs), directly extracting these properties is more challenging since they do not process inputs in the same integrated manner as AlphaFold2. Nevertheless, for MLMs to excel with millions of protein sequences, they must implicitly learn some of these properties, albeit not as explicitly as AlphaFold2. Consider a scenario where an amino acid, engaged in a long-range contact with another, is masked. This situation implies the amino acids must have opposite charges; if one is positively charged, the masked one must be negatively charged to facilitate contact. This requirement narrows down the selection of possible amino acids, thus reducing the set of amino acid to choose from, i.e. lower perplexity.

<figure style="width:35%;">
  <img src="/images/mlm_intuition.png" alt="AlphaFold2 Architecture">
  <figcaption style="text-align: center;">Fig. 6 <b>(own creation)</b></figcaption>
</figure>


### From Millions to Billions Parameters: Understanding the Impact of Scale
<figure>
  <img src="/images/Scale.jpg" alt="Scale Figure 1">
  <figcaption style="text-align: center;">Fig. 7 <b>(Source: from paper)</b></figcaption>
</figure>
From left to right shows the model starting from 8M parameters to 15B parameters. On the x-axis the smaller model is depicted and on the y-axis the next bigger one. 
Visible from the plots, there is a trend of improving perplexity for almost all proteins shown by the blueish color. Simultaneously, the unsupervised contact precision improves as well for most of the proteins with scale. This showcases that the perplexity is indeed highly correlated with the contact precision. 

Interestingly, in the third and forth plot there are proteins, albeit just a few of them, were the performance decreases a lot. This was not further investigated by the authors, but I can give you a possible high-level explanation to that. 

When a model is scaled up, it learns more complex information, improving performance on most data points in a dataset. However, some data points, like certain proteins in this context, may drastically underperform with larger models. This can happen because these outliers differ significantly from the majority of data. Smaller models might do better with such outliers because they focus on basic, common characteristics found across the dataset. Imagine this scenario with the English and German languages, where most data is in German. At a basic level, both languages follow the simple sentence structure of Subject + Verb + Object. For example, in English, we say "I read books," and in German, "Ich lese Bücher."

This foundational structure is easily captured by smaller models. However, when sentences become more complex, such as those including a modal verb, the structure in German shifts to Subject + (modal verb) + Object + Verb, as in "Ich muss Bücher lesen." However, in English, the sentence structure doesn't change, even with the addition of a modal verb, as seen in "I must read books." This means that smaller models, which focus on basic structures, continue to perform well with these slightly more complex English sentences. If a model is trained mainly on German data and then scaled up, it learns to adjust for the modal verb placement, which negatively affect the performance on the rare English sentences with modal verbs. The scaled-up model is now better at handling complex German sentence structures, i.e. for sentences with modal verbs change structure from: "Subject + Verb + Object" to "Subject + (Modal Verb) + Object + Verb". However, applying this knowledge to the rare English sentences with modal verbs would in turn lead to a performance loss as the structure still is the same even with modal verbs for English.

Coming back to the proteins that perform worse, we can't say for sure why it happens, but what should be clear is that these proteins should differ a lot from the general data and I think it is important to analyse such cases further to make sure that the model always gives a good prediction no matter which protein we want to infer.
<figure>
  <img src="/images/Scale2.jpg" alt="Scale Figure 2">
  <figcaption style="text-align: center;">Fig. 8 <b>(animated and own creation)</b></figcaption>
</figure>

The figure illustrates the effect of scaling the ESM-2 model on the prediction accuracy for two different proteins, utilizing the ESMFold extension. Alongside perplexity, two additional measures are introduced: RMSD, which represents the root-mean-square deviation of atomic positions, indicating that a lower value suggests better alignment of predicted atom positions with the actual structure; and pLDDT, a confidence score where a higher percentage indicates greater model certainty in the predicted alignment with the true protein structure.

From the visual representation of the predicted proteins, it is apparent that as ESM-2 is scaled up, RMSD decreases while the pLDDT confidence score increases. This transition is depicted by the change in color from pinkish color (denoting lower confidence) to bluish color (denoting higher confidence). The ground truth structure of the protein is shown in gray, and it is evident that as the model scales, the predicted regions increasingly converge with the gray areas, indicating greater accuracy in the model’s predictions. Simultaneously, the perplexity decreases with scale as well confirming again that MLM perplexity is a good measure for the overall performance of structure prediction.


### Comparison to SOTA Models: AlphaFold-2 and RosettaFold
<figure>
  <img src="/images/Comparison.jpg" alt="Comparison">
  <figcaption style="text-align: center;">Fig. 9 <b>(animated and own creation)</b></figcaption>
</figure>
As can be seen from the above figure in the left bar chart, ESMFold yields competitive results on CAMEO but fails to do so on CASP14. This discrepancy could be explained by the competitive nature of CASP14, meaning that the amino acid sequences we have to infer the structure from are highly complex.

The authors also conducted an ablation study where where no evolutionarily related sequences (MSA) were used in the RosettaFold and AlphaFold2 models. Here, ESMFold significantly outperformed the state-of-the-art (SOTA) models. However, it's important to note that this comparison may not be entirely fair, as the SOTA models aren't specifically trained to perform without MSA input.

On the right, the scatter plots illustrate that ESMFold maintains consistent performance on proteins with low perplexity scores across both datasets, matching the accuracy of predictions regardless of the dataset complexity.


## Scientific Contribution: ESM Metagenomic Atlas
The ESMFold's accelerated protein folding capability and its independence from evolutionary related patterns have resulted in the creation of the first metagenomic atlas at an unprecedented scale, comprising hundreds of millions of protein structures. For context, there have been only 200,000 protein structures experimentally determined to date, and AlphaFold2's database has accumulated around 200 million predicted structures over three years. In contrast, the ESM Metagenomic Atlas boasts over 600 million protein structures, all predicted within a mere two weeks.

This remarkable achievement marks the first occasion that metagenomic proteins—those from microorganisms in environments like soil and seawater—have been folded on such a scale. ESMFold has achieved the prediction of unique protein structures that lacked any previously determined experimental counterparts. This success comes from its ability to predict protein structures independently of genetic database searches and therefore ESMFold is able to explore the protein landscape distant from existing knowledge. Consequently, an impressive number of over 113 million proteins from the metagenomic atlas have been classified with very high confidence.

The Metagenomic Atlas opens up new frontiers in scientific research, uncovering novel protein structures that coould play an important role across various fields. These structures could be crucial in drug discovery, where understanding protein folding can lead to the development of new pharmaceuticals. They could also be pivotal in the advancement of bioengineering, providing insights into enzyme design for industrial processes, or in agriculture, aiding the creation of disease-resistant crops. Moreover, the atlas's potential applications in environmental science are significant, such as in bioremediation strategies where specific protein structures may help degrade pollutants.

All the predicted structures from this monumental effort are freely accessible in the ESM Metagenomic Atlas, available through a user-friendly web interface, API, and bulk download options. This open science resource facilitates both broad and detailed analyses, enabling researchers worldwide to explore the full breadth of the hundreds of millions of predicted structures.

If you want to find more about ESM Metagenomic Atlas, I can recommend you to read through the official blog post from Meta themselves: <a href="https://ai.meta.com/blog/protein-folding-esmfold-metagenomics">https://ai.meta.com/blog/protein-folding-esmfold-metagenomics</a>.

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
