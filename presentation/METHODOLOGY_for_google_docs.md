Suggested Google Doc title: How We Did It: The Method, Explained Step by Step


HOW WE DID IT: THE METHOD, EXPLAINED

This explains our method from start to finish, in plain language, with the reason
behind each choice. The figure folder is presentation/story_figures.

[ Figure 0 here. File: figure_0_method.png.
  Caption: The whole method on one page. Audio goes through a frozen model, we keep the
  per frame activations, an SAE turns them into features, we pick the real emotion features
  with three filters, then we label them and run five analyses. ]


1. The goal

We want to know, inside these speech models, whether emotion is carried by the tone of
voice or by the words, and where in the model this happens. Not just "can a classifier
read emotion," but "which internal pieces carry it, and what do they react to."


2. What we started from (the prior work)

The earlier probing project already built the parts we needed. It runs each of the six
frozen models, takes the hidden state of every layer, averages it over time into one vector
per clip (25 vectors per clip, one per layer), and trains a simple logistic regression probe
on each layer with 5 fold cross validation. From this it learned where emotion is readable
and produced four results we now want to explain: a middle to late peak in accuracy by depth,
a collapse in English to Chinese transfer, a drop when testing on unseen speakers (MESD), and
a per model "text bias" on mismatched speech (Whisper resists, the others do not).

We keep all of this. Same models, same layers, same probe, same datasets, same saved
activations. The only thing we add is a way to look inside a layer instead of treating it as
a black box. That tool is the Sparse Autoencoder.


3. The core tool: the Sparse Autoencoder (SAE)

A layer's activation is one long vector of 1024 numbers. It is dense, meaning every number is
a little bit of everything, so you cannot point at "the emotion part." An SAE fixes this. It
learns to rewrite that vector using a big dictionary of 8192 simple parts, where only 32 are
active at a time. Each active part is a "feature." Because so few are active, each feature
tends to mean one thing. This is the move from a tangled signal to a labelled fuse box.

One important detail. The probe averages each layer over time, which gives only one vector per
clip. That is far too few examples to train a dictionary of 8192 features. So we re run the same
frozen models and keep the activations un pooled: one vector per short audio frame. One layer of
CREMA-D alone gives about 940 thousand frames. We train the SAE on those.

We check that the SAE is trustworthy before using it: it rebuilds the original activation to 93
percent, and almost none of its features are dead. So it is a faithful description of the layer,
not a lossy guess.


4. The experimental design (why these datasets)

This is the part that makes the conclusions valid.

CREMA-D is our main dataset because the 91 actors all say the same 12 sentences in 6 emotions.
The words are held fixed, so if a feature tracks emotion here, it cannot be tracking the words.
It has to be tracking the voice. This is a built in control.

The confound check is the second control, and it is the heart of the method. Even on CREMA-D,
many features react to the specific sounds of the words (phonetics) or to the specific speaker.
Those are not emotion features, even if they correlate with emotion. So we require a real emotion
feature to be more informative about the emotion than about the sentence or the speaker. Without
this check, the model looks like it has hundreds of emotion features, but most are really phoneme
or speaker detectors. This is the classic trap, and it is why earlier intuition was unclear.

EMIS is our third dataset, used at the end. It is synthetic speech where the written emotion and
the spoken emotion are set to be different on purpose. This is the only way to directly catch a
model using the words instead of the voice.


5. How we pick the emotion features (three filters)

We look at every one of the 8192 features and keep it only if it passes all three:

Filter 1, significance. We measure how much the feature's firing (on or off) shares information
with the emotion label, using mutual information. We compare this to what you would get by chance
by shuffling the labels, and keep only features that are clearly above chance (5 standard
deviations above the shuffled baseline).

Filter 2, monosemanticity. We require that at least half of the feature's activity falls on a
single emotion. This removes features that mix several emotions together.

Filter 3, the confound check. We require that the feature shares more information with the emotion
than with the sentence, and more than with the speaker. This removes phoneme and speaker detectors.

Filters 1 and 2 alone leave 394 features. Adding filter 3 brings it down to 61. Those 61 are our
clean emotion features. They are mostly anger and sadness, the emotions with the clearest voice
signature.


6. How we label a feature as acoustic or lexical

For each of the 61 features we ask a simple question: is its firing better predicted by the tone
of the voice, or by which sentence was spoken? For the tone we use eGeMAPS, a standard set of
acoustic measurements like pitch, loudness, and voice quality. For the words we use the sentence
identity. We train a small classifier each way and compare how well each predicts the feature's
firing, using AUC (we use AUC and not a fit score because the features are silent about 96 percent
of the time, which breaks a normal fit score). On CREMA-D, where the answer must be acoustic, 60 of
the 61 come out acoustic. This confirms the labelling method works, so we can trust it elsewhere.


7. The five analyses

Each analysis is a clear question with a clear rule for the answer.

Depth. We repeat the SAE and the feature picking at all 25 layers. We compare the layer where the
probe is most accurate to the layer where there are the most clean emotion features. If they are
different, then being readable and being well organized are different things. (They are: accuracy
peaks at layer 10, features peak at layer 7.)

Cross model. For each pair of models we describe each emotion feature by how it fires across the
shared clips, and we count how many of one model's features have a close match in the other. High
overlap means the models use a shared mechanism.

Cross speaker. We classify emotion using only the 61 features, once normally and once with leave
one speaker out, so no speaker appears in both training and testing, across all 91 actors. A small
drop means the features are about the emotion, not the person.

Causal sufficiency. We try to decode emotion from only the 61 acoustic features, then from 61
lexical features, then from 61 random features. Whichever subset keeps the accuracy is the one that
actually carries emotion. (Acoustic 0.55, lexical 0.43, random 0.42, so acoustic is sufficient and
lexical is not.)

EMIS. We train the emotion probe on the matched clips and test on the mismatched ones, and we check
whether the prediction follows the voice or the words. We define text bias as the accuracy toward
the words minus the accuracy toward the voice, and we measure it at every layer on the raw
activations. Positive means the model is using the words. We find this only in the deep layers of the
speech recognition models. Whisper never shows it.

One caution we learned here. For this last test we must use the raw model activations, not the SAE's
rebuilt version. Running the test on the rebuilt version flips Whisper's result by mistake. The SAE
is the right tool for reading features, but not as a stand in for the raw signal in this one probe.


8. Why the conclusions are trustworthy

Three controls hold the method together. The fixed sentences of CREMA-D remove the words as an
explanation. The confound check removes phonetics and speaker as explanations. And the acoustic
versus lexical labelling has a positive control (it must say acoustic on CREMA-D, and it does). Only
after these do we make the acoustic claim, and only EMIS, where words and voice are split on purpose,
lets us speak about the word shortcut directly.
