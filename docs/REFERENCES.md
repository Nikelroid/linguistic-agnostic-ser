# References

The PDFs themselves are not redistributed here. Each entry links to the
publisher or preprint so you can obtain it under your own access.

## Directly load-bearing

**Probing Speech Emotion Recognition Transformers for Linguistic Knowledge.**
Triantafyllopoulos, Wagner, Wierstorf, Schuller et al., Interspeech 2022.
<https://arxiv.org/abs/2204.00400>
The paper this project set out to test. It reports that transformer SER models
carry linguistic information, which is the question our EMIS and sparse
autoencoder experiments answer mechanistically.

**Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence
Gap.** Wagner, Triantafyllopoulos, Wierstorf, Schuller et al., IEEE TPAMI 2023.
<https://arxiv.org/abs/2203.07378>
Establishes the valence gap we reproduce in the dimensional track, where valence
is consistently the hardest of the three targets.

**Disentangling Prosody Representations With Unsupervised Speech
Reconstruction.** Qu, Li, Busso et al., IEEE/ACM TASLP 2023.
<https://arxiv.org/abs/2212.06972>
Background for separating prosodic from lexical content, which is what the
sparse autoencoder confound control does by a different route.

## Corpora

RAVDESS <https://doi.org/10.1371/journal.pone.0196391> ·
EmoDB <http://emodb.bilderbar.info> ·
IEMOCAP <https://sail.usc.edu/iemocap/> ·
SAVEE <http://kahlan.eps.surrey.ac.uk/savee/> ·
AESDD <https://mlearn.cmb.uoa.gr/aesdd/> ·
MESD <https://doi.org/10.17632/cy34mh68j9.5> ·
CREMA-D <https://github.com/CheyneyComputerScience/CREMA-D> ·
MSP-Podcast <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html> ·
ESC-50 <https://github.com/karolpiczak/ESC-50>

## Encoders

wav2vec 2.0 <https://arxiv.org/abs/2006.11477> ·
HuBERT <https://arxiv.org/abs/2106.07447> ·
WavLM <https://arxiv.org/abs/2110.13900> ·
Whisper <https://arxiv.org/abs/2212.04356> ·
w2v-BERT <https://arxiv.org/abs/2108.06209> ·
MERT <https://arxiv.org/abs/2306.00107>

## Method

Sparse autoencoders and monosemanticity <https://transformer-circuits.pub/2023/monosemantic-features> ·
TopK sparse autoencoders <https://arxiv.org/abs/2406.04093> ·
LoRA <https://arxiv.org/abs/2106.09685> ·
Gradient reversal for domain adaptation <https://arxiv.org/abs/1409.7495> ·
Centered kernel alignment <https://arxiv.org/abs/1905.00414>
