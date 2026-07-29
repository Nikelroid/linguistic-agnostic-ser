Suggested Google Doc title: Looking Inside Speech Emotion Models: Method and Findings


LOOKING INSIDE SPEECH EMOTION MODELS

A short, plain walk through how we did this and what we found. Figures are gathered in
the folder presentation/story_figures, named to match the placeholders below.


The question

AI models like HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, and Whisper are very good at
guessing emotion from a voice clip. But they were built to turn speech into text. So when
one of them says "anger," is it hearing the angry tone of voice, or is it reading the angry
words? A model that leans on the words will fail on sarcasm, accents, and new languages. We
wanted to find out which one it is, and where inside the model it happens.


How we did it

We started from earlier work that already built the parts we needed. It runs each of the six
frozen models, takes the hidden state of every layer, averages it over time into one vector
per clip, and trains a simple classifier on each layer. From this it learned where emotion is
readable. But that only tells us where, not why, and it treats each layer as a black box. We
keep all of this (same models, same layers, same classifier, same datasets) and add one thing:
a way to look inside a layer.

That tool is a Sparse Autoencoder (SAE). A layer's activation is a long list of 1024 numbers,
and every number is a little bit of everything, so you cannot point at the emotion part. The SAE
rewrites that list using a big dictionary of 8192 simple parts, where only 32 are active at a
time. Each active part is a "feature," and because so few are active, each feature tends to mean
one thing. This turns a tangled signal into a labelled fuse box. One detail: the averaged vectors
are far too few to train this dictionary, so we re run the models and keep the activations frame
by frame instead of averaged. One layer of CREMA-D gives about 940 thousand frames. We checked
that the SAE is faithful before trusting it: it rebuilds the original signal to 93 percent.

Three design choices make the conclusions valid.

First, CREMA-D. Its 91 actors all say the same 12 sentences in 6 emotions. Because the words never
change, a feature that tracks emotion here has to be tracking the voice, not the words. That is a
built in control.

Second, the confound check, which is the heart of the method. Even on CREMA-D, many features react
to the specific sounds of the words or to the specific speaker. Those are not emotion features. So
we keep a feature only if it tells us more about the emotion than about the sentence or the speaker.
We apply three filters in total: the feature must (1) clearly relate to emotion, (2) fire for mostly
one emotion, and (3) beat the sentence and the speaker. Without the third filter the model looks like
it has 394 emotion features, but most are really phoneme or speaker detectors. With it, 61 clean
emotion features remain. This is the step that made the earlier picture confusing, and fixing it is
what made everything clear.

Third, to label each feature, we ask whether its firing is better predicted by the tone of voice
(standard acoustic measurements like pitch and loudness) or by which sentence was spoken. On CREMA-D
the answer must be acoustic, and 60 of the 61 come out acoustic. That positive result tells us the
labelling method works, so we can trust it elsewhere.

[ Figure 0 here. File: figure_0_method.png.
  Caption: The whole method on one page. Audio goes through a frozen model, we keep the per frame
  activations, an SAE turns them into features, we keep only the real emotion features with three
  filters, then we label them and run the analyses below. ]


What we found

1. The model organizes emotion early, but reads it best later.
We did the SAE and the feature picking at all 25 layers, and compared the layer with the most clean
emotion features to the layer where the classifier is most accurate. They are different: the features
peak early (around layer 7) but accuracy peaks deeper (around layer 10). So being easy to read out and
being neatly organized are two different things.

[ Figure 1 here. File: figure_1_depth.png.
  Caption: Blue line is how well a classifier reads emotion (peaks deep, layer 10). Red line is the
  number of clean emotion features (peaks early, layer 7). ]

2. It is the tone of voice, not the words.
60 of the 61 features track the voice, not the words. To be sure, we ran a direct test: we tried to
read emotion from only the acoustic features, then from only the word based features, then from random
features. Acoustic worked (0.55), word based barely worked (0.43), about the same as random (0.42). So
the acoustic features are enough and the word features are not.

[ Figure 2 here. File: figure_2_acoustic_vs_words.png.
  Caption: Each dot is one emotion feature. Almost all sit on the acoustic side, meaning they track the
  voice, not the words. ]

[ Figure 3 here. File: figure_3_causal_test.png.
  Caption: Emotion reads out well from acoustic features, poorly from word features. ]

3. The six models agree, and they ignore the speaker.
We matched features across the models and found they land on a similar set. We also tested on speakers
never seen in training, and accuracy barely dropped. So the features are about the emotion, not about
the person speaking.

[ Figure 4 here. File: figure_4_across_models.png.
  Caption: Left, how much each model's emotion features overlap with the others. Right, how many clean
  emotion features each model has. ]

4. Only some models cheat with words, and only deep down.
We made a test set (EMIS) where the written emotion and the spoken emotion are different on purpose. We
trained on matched clips and tested on mismatched ones, and checked whether each model follows the voice
or the words, layer by layer. The speech recognition models (HuBERT, WavLM, wav2vec2) start trusting the
words, but only in their deep layers. Whisper never does. It stays grounded in the voice everywhere.

[ Figure 5 here. File: figure_5_cheating_test.png.
  Caption: Higher means the model is using the words instead of the voice. The shortcut appears only in
  the deep layers of the speech recognition models. Whisper, in red, stays low everywhere. ]


How this compares to other recent work

We checked the 2025 to 2026 literature. The same SAE tool was used very recently (a paper called AudioSAE),
but for general audio and only as one of several quick tests, without our confound check, our fixed sentence
design, or the tone versus words question. Other recent papers use SAEs on speech to find sounds or singing
style, not emotion. The closest emotion paper studies Whisper but uses older tools, not SAEs. So our specific
combination looks new.


Why this matters

Decoding is not the same as understanding. The emotion these models rely on lives in the voice, it is shared
across models, and it does not depend on the speaker. And when a shortcut through the words is available, only
the speech recognition models take it, and only deep inside, while Whisper does not. This tells us which models
and which layers to trust when prosody and meaning disagree.

[ Figure 6 here (optional, the big picture). File: figure_6_big_picture.png.
  Caption: All six results in one view. ]


What is next

A real transcript test on IEMOCAP, a cross language study to explain why English to Chinese transfer fails, and
a steering test where we turn an emotion feature up to change the model's answer. The code and stubs for these
are ready.


The numbers in one place

Faithful reconstruction by the SAE: 93 percent of the signal
Real emotion features after the three filters: 61 (from 394 candidates)
Emotion features that are acoustic, not word based: 60 of 61
Emotion organized earliest / read best at layer: 7 / 10
Read emotion from acoustic / word / random features: 0.55 / 0.43 / 0.42
Accuracy drop on unseen speakers: about zero
Whisper "uses the words" score on mismatched speech: minus 0.11 (resists)
HuBERT, WavLM, wav2vec2 in deep layers: plus 0.7 to 0.9 (use the words)


Where to find everything

Code is in src/sae. Results, tables, and plots are in results/SAE. The 4 page report, the slides, and the full
technical brief are in the presentation folder. Everything is on the SAE branch.
