Suggested Google Doc title: Looking Inside Speech Emotion Models: What We Found


LOOKING INSIDE SPEECH EMOTION MODELS

A short, plain walk through our work. Figures are gathered in the folder
presentation/story_figures, named to match the placeholders below.


The question

AI models like HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, and Whisper are very good
at guessing emotion from a voice clip. But they were built to turn speech into text.
So when one of them says "anger," is it hearing the angry tone of voice, or is it
reading the angry words? A model that leans on the words will fail on sarcasm,
accents, and new languages. We wanted to find out which one it is.


Our idea

The usual approach, called probing, only tells us where in the model emotion becomes
readable. It cannot tell us which internal pieces carry emotion, or what they react to.
So we used a Sparse Autoencoder (SAE). It untangles a model's internal signal into a
large set of simple, single meaning "features," like turning a messy ball of wires into
a labelled fuse box. Then we can point at the features that carry emotion.

Two choices made this work. First, we test on CREMA-D, where 91 actors say the same 12
sentences in 6 emotions. Because the words never change, any emotion the model finds
there must come from the voice. Second, we added a confound check: we only keep a feature
as an emotion feature if it tells us more about the emotion than about the sentence or
the speaker. Without this check, most "emotion" features are really just detecting the
sounds of letters. After the check, 394 candidate features become 61 real ones.


What we found

1. The model organizes emotion early, but reads it best later.
The clean emotion features are strongest in the early middle layers (around layer 7).
But a simple classifier scores highest deeper (around layer 10). So being easy to read
out and being neatly organized are two different things.

[ Figure 1 here. File: figure_1_depth.png.
  Caption: Blue line is how well a probe reads emotion (peaks deep, layer 10). Red line is
  the number of clean emotion features (peaks early, layer 7). ]

2. It is the tone of voice, not the words.
60 of the 61 emotion features are explained by acoustic properties like pitch and loudness,
not by which sentence was spoken. To be sure, we ran a direct test: emotion can be decoded
from the acoustic features alone (score 0.55), but barely from the word based features (0.43),
which is about the same as random (0.42).

[ Figure 2 here. File: figure_2_acoustic_vs_words.png.
  Caption: Each dot is one emotion feature. Almost all sit on the acoustic side, meaning they
  track the voice, not the words. ]

[ Figure 3 here. File: figure_3_causal_test.png.
  Caption: Emotion decodes well from acoustic features, poorly from word features. ]

3. The six models agree, and they ignore the speaker.
The models land on a similar set of acoustic emotion features. They also keep working on
speakers they never heard in training, with almost no drop in accuracy. So the features are
about the emotion, not about the person speaking.

[ Figure 4 here. File: figure_4_across_models.png.
  Caption: Left, how much each model's emotion features overlap with the others. Right, how many
  clean emotion features each model has. ]

4. Only some models cheat with words, and only deep down.
We made a special test set (EMIS) where the written emotion and the spoken emotion are different
on purpose. The speech recognition models (HuBERT, WavLM, wav2vec2) start trusting the words,
but only in their deep layers. Whisper never does. It stays grounded in the voice at every layer.

[ Figure 5 here. File: figure_5_cheating_test.png.
  Caption: Higher means the model is using the words instead of the voice. The shortcut appears
  only in the deep layers of the speech recognition models. Whisper, in red, stays low everywhere. ]


How this compares to other recent work

We checked the 2025 to 2026 literature. The same SAE tool was used very recently (a paper called
AudioSAE), but for general audio and only as one of several quick tests, without our confound check,
our fixed sentence design, or the tone versus words question. Other recent papers use SAEs on speech
to find sounds or singing style, not emotion. The closest emotion paper studies Whisper but uses older
tools, not SAEs. So our specific combination looks new.


Why this matters

Decoding is not the same as understanding. The emotion these models rely on lives in the voice,
it is shared across models, and it does not depend on the speaker. And when a shortcut through the
words is available, only the speech recognition models take it, and only deep inside, while Whisper
does not. This tells us which models and which layers to trust when prosody and meaning disagree.

[ Figure 6 here (optional, the big picture). File: figure_6_big_picture.png.
  Caption: All six results in one view. ]


What is next

A real transcript test on IEMOCAP, a cross language study to explain why English to Chinese transfer
fails, and a steering test where we turn an emotion feature up to change the model's answer. The code
and stubs for these are ready.


The numbers in one place

Faithful reconstruction by the SAE: 93 percent of the signal
Real emotion features after the confound check: 61 (from 394 candidates)
Emotion features that are acoustic, not word based: 60 of 61
Emotion organized earliest / read best at layer: 7 / 10
Decode emotion from acoustic / word / random features: 0.55 / 0.43 / 0.42
Accuracy drop on unseen speakers: about zero
Whisper "uses the words" score on mismatched speech: minus 0.11 (resists)
HuBERT, WavLM, wav2vec2 in deep layers: plus 0.7 to 0.9 (use the words)


Where to find everything

Code is in src/sae. Results, tables, and plots are in results/SAE. The 4 page report, the slides,
and the full technical brief are in the presentation folder. Everything is on the SAE branch.
