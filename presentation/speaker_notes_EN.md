# Speaker notes — SAE Dissection of Speech Emotion Recognition (English)

Pacing: ~10 slides, ~10–12 minutes. One slide ≈ one idea. Numbers to memorize are **bold**.

---

## Slide 1 — Title
"This is a mechanistic-interpretability study of speech emotion recognition. The one-line story: we use sparse autoencoders to go from *what a probe can read out* of a frozen speech model to *which internal features actually carry emotion* — and whether that signal is acoustic or lexical."

---

## Slide 2 — The question
"Frozen speech encoders — HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, and Whisper — top emotion-recognition benchmarks. But they were pretrained for speech recognition, so they carry a lot of *text* information. Probing tells us *where* in the network emotion becomes readable, but it can't tell us three things we care about: which features encode emotion, whether the model is using acoustics (how something is said) or lexical content (what is said), and whether the story is the same across different encoders and speakers. Our idea is to use sparse autoencoders to answer the *mechanism* question, not just decodability."

Anticipated question — "Why does acoustic vs lexical matter?" → "Because a model that reads emotion off the words will break on sarcasm, accented speech, and low-resource languages where prosody and meaning disagree."

---

## Slide 3 — Method in one slide
"We train a TopK sparse autoencoder — 8× over-complete, 32 active features per frame — on the *per-frame* activations of each frozen encoder, about 940 thousand frames. It reconstructs faithfully: 93% of variance explained, basically no dead features. The key design choice is the corpus: CREMA-D has 91 actors saying 12 *fixed* sentences in 6 emotions. Because the text is held constant, any emotion signal we recover there must be acoustic — that's a built-in positive control. The second key idea is *confound control*: we only call a feature an 'emotion feature' if its firing tells us more about emotion than about the sentence (which captures phonetics) or the speaker. Without that control, about two-thirds of the so-called emotion features are really phonetic or speaker features — that's the classic trap where an SAE just rediscovers phonemes instead of affect. After control we keep **61** clean features, dominated by anger and sadness."

---

## Slide 4 — Q1: depth, decoding ≠ disentangling
"First result, about depth. If you train a linear probe, accuracy peaks **deep**, around layer 10, and stays high. But if you ask where the *clean, monosemantic* emotion features are, that peaks **early** — layer 7 for the count, layer 2 for monosemanticity. In the deep layers the model is still very decodable, but the emotion has become *distributed* — it's smeared across directions rather than packed into a few clean features. So the headline is: emotion gets *organised* into interpretable features earlier than the depth where a probe reads it best. Decodability and disentanglement are different axes."

---

## Slide 5 — Q2: across encoders and speakers
"Second, does this generalize. All six encoders have healthy emotion-feature populations, all dominated by anger and sadness. When we match features across encoders by how they fire, Whisper's overlap with the SSL models is the same as the SSL models' overlap with each other — so on fixed-lexicon speech Whisper is *not* special. Its known difference, which we'll see in a moment, is about text, not acoustics. And the features are speaker-invariant: under leave-one-speaker-out across all 91 actors, accuracy basically doesn't drop, and each feature fires for about 95% of the actors who express its emotion. That also explains a known result — the big speaker-out collapse on the MESD dataset comes from *speaker-entangled* features, exactly the ones our confound control throws away."

---

## Slide 6 — Q3: acoustic, not lexical
"Third, is the signal acoustic or lexical. The positive control says acoustic: 60 of 61 features are better predicted by eGeMAPS acoustic descriptors than by sentence identity. And causally — if we let emotion be decoded only from the acoustic features, we get **0.55**; only from the lexical features, **0.43**; from random features, **0.42**. So the acoustic features are sufficient and the lexical ones aren't. What the probe exploits is paralinguistic delivery, not words."

---

## Slide 7 — Q4: EMIS, the lexical shortcut is a deep-layer thing
"Fourth, the lexical shortcut. EMIS is synthetic speech where the *text* emotion and the *voice* emotion are deliberately set to different things. We train on the congruent clips and test on the incongruent ones, and measure text-bias: how much the model predicts the text emotion instead of the voice emotion. The new finding is that the shortcut is a *deep-layer* phenomenon — it shows up only in the deep layers of the ASR-trained SSL encoders, HuBERT, WavLM, wav2vec2, at plus 0.7 to 0.9. Whisper stays audio-grounded at *every* layer, slightly negative even at its last layer. MERT and w2v-BERT never take it. This reproduces the published per-encoder text-bias numbers and adds the depth story on top."

Important caveat to say out loud: "One methodological note — if you run this probe on the SAE *reconstruction* instead of the raw activations, it spuriously flips Whisper. So for this particular test we use raw activations; the SAE is the right tool for the per-feature analysis but not as a stand-in here."

---

## Slide 8 — Where this sits in the literature
"On positioning: the closest method paper is AudioSAE at EACL 2026 — same SAE recipe, but on two encoders, and emotion is just one of four generic probe tasks. They do no emotion-feature interpretation, no confound control, no factorial corpus, no acoustic-versus-lexical split. The other SAE-on-speech papers find phonetic or singing-technique features, not affect, and don't do a depth sweep. The closest mechanistic SER paper uses probing and CKA but explicitly no SAEs. So our core combination — confound-controlled SAE emotion features on a lexicon-controlled factorial corpus across six encoders, plus the depth and EMIS findings — is, as far as we can tell, un-scooped."

---

## Slide 9 — Limitations and future work
"Honest limitations: because the signal is distributed, removing a few features doesn't hurt much, so we rely on the sufficiency test rather than ablation. We must not substitute the SAE reconstruction for raw activations in the EMIS probe. And the cross-encoder and EMIS arms use one layer per encoder, not the full 25-layer grid for all six. Future work is mostly ready as stubs: an IEMOCAP causal ablation with real transcripts, a SpeechCraft cross-lingual study to explain the English–Chinese collapse, feature steering, and dimensional valence/arousal/dominance targets."

---

## Slide 10 — Takeaways
"Four things to remember: one, SAEs let us move SER from decodability to mechanism, and confound control is what makes it honest. Two, decoding is not disentangling — emotion organises early but decodes best deep. Three, the emotion features are acoustic and speaker-invariant, and shared across all six encoders. Four, the lexical shortcut is a deep-layer SSL phenomenon, and Whisper resists it everywhere. Everything — code, data pointers, all artifacts, and the report — is on the SAE branch. Thank you."

---

### Likely Q&A
- **"Why CREMA-D and not IEMOCAP?"** CREMA-D's fixed lexicon is the control that lets us *separate* acoustic from lexical. IEMOCAP has real transcripts, which is exactly why it's our planned causal-ablation follow-up.
- **"Is 61 features too few?"** It's after a deliberately strict triple filter (significant + monosemantic + beats both confounds). The point is purity, not count; the looser set of 394 is dominated by phonetics.
- **"Does this depend on the SAE hyperparameters?"** Reconstruction is stable (93% variance, ~0 dead). The qualitative depth and acoustic findings are robust; we haven't swept k/expansion exhaustively — listed as future work.
- **"How is this different from AudioSAE?"** Same tool, different question: they probe four generic attributes on two models; we do confound-controlled affect-feature interpretation on six, plus the depth and EMIS-by-depth results.
