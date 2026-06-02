#!/usr/bin/env python
"""Generate a single self-contained ENGLISH briefing for the SAE study.
Complete walk-through in plain language: motivation, key ideas, the full process
(Steps 0-8), findings, figures, literature review, speaker notes, and Q&A.
Embeds figures as base64 so the file is portable (drop on any GitHub page)."""
import base64, os

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "latex", "images")


def b64(name):
    p = os.path.join(IMG, name)
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""


F = {n: b64(f"fig_{n}.png") for n in
     ["summary", "depth", "emis", "sufficiency", "acoustic", "crossenc", "emotionfeats"]}

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAE Dissection of Speech Emotion Recognition — Team Briefing</title>
<style>
  :root {{ --navy:#1f3a6e; --red:#c0392b; --ink:#222; --soft:#f5f7fc; --line:#dde3ee; --green:#1e7d4f; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
         line-height:1.65; margin:0; background:#fff; }}
  header {{ background:var(--navy); color:#fff; padding:30px 20px; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:0 20px; }}
  header h1 {{ font-size:27px; margin:0 0 6px; }}
  header .sub {{ opacity:.9; font-size:16px; }}
  header .authors {{ font-size:13px; opacity:.8; margin-top:10px; }}
  main {{ max-width:880px; margin:0 auto; padding:10px 20px 90px; }}
  h2 {{ color:var(--navy); border-bottom:2px solid var(--line); padding-bottom:6px; margin-top:42px; font-size:23px; }}
  h3 {{ color:var(--navy); margin:22px 0 4px; font-size:18px; }}
  nav {{ position:sticky; top:0; background:rgba(255,255,255,.97); border-bottom:1px solid var(--line);
        padding:10px 0; z-index:10; font-size:13.5px; }}
  nav .wrap {{ display:flex; flex-wrap:wrap; gap:6px 14px; }}
  nav a {{ color:var(--navy); text-decoration:none; }}
  nav a:hover {{ text-decoration:underline; }}
  .tldr {{ background:var(--soft); border:1px solid var(--line); border-left:5px solid var(--navy);
          border-radius:8px; padding:14px 18px; margin:18px 0; }}
  .card {{ background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:6px 18px; margin:14px 0; }}
  .finding {{ border-left:4px solid var(--navy); padding:2px 0 2px 16px; margin:18px 0; }}
  .step {{ border:1px solid var(--line); border-radius:9px; padding:8px 16px; margin:12px 0; }}
  .step h3 {{ margin-top:8px; }}
  .step .why {{ color:#555; font-size:14.5px; }}
  table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:14px; }}
  th,td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }}
  th {{ background:var(--navy); color:#fff; }}
  tr:nth-child(even) td {{ background:var(--soft); }}
  .yes {{ color:var(--green); font-weight:bold; }} .no {{ color:var(--red); font-weight:bold; }}
  figure {{ margin:20px 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; }}
  figcaption {{ font-size:13.5px; color:#555; margin-top:7px; }}
  .pt {{ background:#fffae6; border:1px solid #efe0a6; border-radius:8px; padding:6px 14px; margin:9px 0; }}
  code {{ background:#eef1f7; padding:1px 6px; border-radius:4px; font-size:90%; }}
  .glossary dt {{ font-weight:bold; color:var(--navy); margin-top:10px; }}
  footer {{ color:#666; font-size:13.5px; border-top:1px solid var(--line); margin-top:46px; padding-top:16px; }}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Opening up Speech-Emotion AI: what's actually inside the models</h1>
  <div class="sub">A complete, plain-language briefing on our Sparse-Autoencoder study &mdash; read this to get ready.</div>
  <div class="authors">Nima Kelidari &middot; Minoo Ahmadi &middot; Chaitanya Parwatkar &middot; Xiangxu (Henry) Lin &nbsp;|&nbsp; USC ICT &mdash; IHP Lab</div>
</div></header>

<nav><div class="wrap">
  <a href="#tldr">TL;DR</a><a href="#question">The question</a><a href="#ideas">Key ideas</a>
  <a href="#process">The whole process</a><a href="#findings">Findings</a><a href="#numbers">Numbers</a>
  <a href="#lit">Recent literature</a><a href="#talk">What to say</a><a href="#qa">Likely questions</a>
  <a href="#limits">Limits &amp; next</a><a href="#where">Where things are</a>
</div></nav>

<main>

<section id="tldr">
<div class="tldr">
<strong>In one paragraph.</strong> AI models that score top marks at guessing emotion from a voice were actually built to <em>transcribe words</em> &mdash; so we never quite knew whether they listen to the <em>tone of voice</em> or to the <em>words</em>, or <em>which</em> internal parts do the work. We used a tool called a <strong>Sparse Autoencoder (SAE)</strong> to open six of these models and read their internal "features" one by one. Using a clever dataset where everyone says the <em>same sentences</em>, we found: (1) the models tidy emotion into clean features <em>early</em> but a probe reads it best <em>deeper</em>; (2) the emotion features are <strong>acoustic</strong> (tone), not lexical (words), and they're shared across models and work on unseen speakers; and (3) when a model <em>can</em> cheat by reading the words, only the speech-recognition models do &mdash; and only in their deep layers; <strong>Whisper never cheats</strong>.
</div>
</section>

<section id="question">
<h2>1. The big question</h2>
<p>Modern speech models (HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, Whisper) are excellent at guessing the emotion in a clip. But they were originally trained to turn speech into text, so they carry a lot of <em>language</em> information inside them. That raises a worry:</p>
<div class="card"><p>When one of these models detects "anger," is it hearing the <strong>angry tone of voice</strong> &mdash; or is it secretly <strong>reading the angry words</strong>? A model that leans on the words will break exactly where we need it most: sarcasm, accents, and languages it wasn't trained on.</p></div>
<p>The usual way to study this ("probing") tells us <em>where</em> in the model emotion becomes readable, but it treats each layer as a black box. It can't tell us <em>which</em> pieces inside encode emotion, or <em>what</em> they react to. That's the gap we fill.</p>
</section>

<section id="ideas">
<h2>2. Key ideas in plain words</h2>
<dl class="glossary">
<dt>Sparse Autoencoder (SAE)</dt><dd>A tool that takes the tangled internal signal of a model and untangles it into a big set of simple, mostly single-meaning "features." Think of turning a messy ball of wires into a labelled fuse-box, so we can point at the few wires that carry "emotion."</dd>
<dt>Monosemantic feature</dt><dd>A feature that means one thing. We want emotion features that fire for one emotion (say, anger) and stay quiet otherwise &mdash; not features that mix several things together.</dd>
<dt>CREMA-D (our main dataset)</dt><dd>91 actors say the <em>same 12 sentences</em> in 6 emotions. Because the words never change, anything the model learns about emotion here must come from the <em>voice</em>, not the text. This is our built-in control.</dd>
<dt>The confound check</dt><dd>The big methodological trick. We only trust a feature as an "emotion feature" if it tells us more about the <em>emotion</em> than about the <em>sentence</em> (which captures the sounds of the words) or the <em>speaker</em>. Without this, the model's "emotion" features are mostly just detecting <em>sounds of letters</em> &mdash; the classic trap where an SAE rediscovers phonemes, not feelings.</dd>
<dt>Decodability vs. disentanglement</dt><dd>"Decodable" = a simple classifier can read emotion out. "Disentangled" = emotion sits in a few clean, separate features. These turn out to be <em>different</em> things, peaking at different depths.</dd>
<dt>Acoustic vs. lexical</dt><dd>Acoustic = <em>how</em> it's said (pitch, loudness, voice quality). Lexical = <em>what</em> is said (the words). The core question of the whole study.</dd>
<dt>EMIS</dt><dd>A special test set of synthetic speech where the <em>written</em> emotion and the <em>spoken</em> emotion are deliberately mismatched &mdash; the perfect way to catch a model "cheating" with the words.</dd>
</dl>
</section>

<section id="process">
<h2>3. The whole process, step by step</h2>
<p>The study runs as a numbered pipeline. Here is what each step does and why, in plain terms. (Each one is a single reproducible command in the code.)</p>

<div class="step"><h3>Step 0 &mdash; Take stock</h3>
<p>We inventoried the model activations already saved on the cluster, to see exactly what we had to work with.</p>
<p class="why">Why: avoid re-computing expensive things; build the loader around what's really on disk.</p></div>

<div class="step"><h3>Step 0.5 &mdash; Get the right data</h3>
<p>We downloaded CREMA-D (the fixed-sentence corpus) and later EMIS (the mismatched-speech corpus), and re-ran each frozen model to save its internal activations <em>frame by frame</em> (not averaged), because an SAE needs lots of examples.</p>
<p class="why">Why: the previously saved activations were averaged to one vector per clip &mdash; far too few to train a big SAE. Per-frame gave us ~940,000 training examples from one layer.</p></div>

<div class="step"><h3>Step 1 &mdash; Train the SAE</h3>
<p>We trained the SAE on those activations. It learns to rebuild the signal using only a few active features at a time.</p>
<p class="why">Why / check: it must rebuild the signal faithfully before we trust its features. It explains <strong>93%</strong> of the signal with almost no wasted ("dead") features.</p></div>

<div class="step"><h3>Step 2 &mdash; Find the emotion features (with the confound check)</h3>
<p>We scored every feature by how much its firing relates to the emotion, then kept only those that are clean (single-emotion) <em>and</em> beat the confound check (more about emotion than about the sentence or speaker).</p>
<p class="why">Result: 394 candidate features &rarr; <strong>61 genuine</strong> emotion features. Two-thirds of the candidates were really about the words or the speaker &mdash; exactly the trap we wanted to avoid.</p></div>

<div class="step"><h3>Step 3 &mdash; Acoustic or lexical? (the sanity check)</h3>
<p>For each emotion feature we asked: is its firing better explained by acoustic measurements (pitch, loudness&hellip;) or by which sentence was spoken?</p>
<p class="why">Why: on fixed-sentence CREMA-D the answer <em>must</em> be acoustic &mdash; and <strong>60 of 61</strong> features came out acoustic. This proves our labelling method works before we trust it elsewhere.</p></div>

<div class="step"><h3>Step 4 &mdash; Depth sweep</h3>
<p>We repeated the analysis at all 25 layers and compared two curves: how well a probe decodes emotion, and how many clean emotion features exist.</p>
<p class="why">Finding: the two peak at <em>different</em> depths (see Finding 1).</p></div>

<div class="step"><h3>Step 5 &mdash; Compare the six models</h3>
<p>We matched features across the six encoders to see if they "think" alike.</p>
<p class="why">Finding: they share a common emotion-feature basis; Whisper is not special here (see Finding 2).</p></div>

<div class="step"><h3>Step 6 &mdash; Test on unseen speakers</h3>
<p>We re-ran the emotion test making sure the people in the test were never in training.</p>
<p class="why">Finding: almost no drop &mdash; the features are about the emotion, not the person.</p></div>

<div class="step"><h3>Step 7 &mdash; The causal test</h3>
<p>We checked whether emotion can be decoded from <em>only</em> the acoustic features vs. <em>only</em> the lexical ones.</p>
<p class="why">Finding: acoustic features are enough (0.55), lexical ones aren't (0.43) &mdash; the signal is acoustic (see Finding 3).</p></div>

<div class="step"><h3>Step 8 &mdash; The "cheating" test (EMIS)</h3>
<p>On the mismatched-speech set, we measured how often each model goes with the <em>words</em> instead of the <em>voice</em>, layer by layer.</p>
<p class="why">Finding: only the speech-recognition models cheat, and only deep down; Whisper never does (see Finding 4).</p></div>
</section>

<section id="findings">
<h2>4. What we found (the four headlines)</h2>
<div class="finding"><h3>1. The model tidies emotion up early, but reads it best later</h3>
<p>Clean, single-meaning emotion features appear in the <strong>early-middle layers (2&ndash;7)</strong>. Yet a simple classifier scores highest <strong>deeper (layer 10)</strong>. So "easy to read off" and "neatly organised" are two different things &mdash; emotion gets organised early, then re-mixed for easy reading later.</p></div>
<div class="finding"><h3>2. All six models share the same emotion features &mdash; and ignore the speaker</h3>
<p>Every model lands on a similar set of acoustic emotion features, and they keep working on speakers never seen in training (almost no accuracy drop). The features are about the emotion, not the person. On these fixed sentences, even Whisper looks like the others &mdash; its famous difference is about words, which this dataset can't show.</p></div>
<div class="finding"><h3>3. It's the tone of voice, not the words</h3>
<p><strong>60 of 61</strong> emotion features are explained by acoustic properties, not by which sentence was spoken. And emotion can be decoded from the acoustic features (0.55) far better than from the lexical ones (0.43, barely above random 0.42).</p></div>
<div class="finding"><h3>4. When a model <em>can</em> cheat with words, only some do &mdash; and only deep down</h3>
<p>On mismatched speech, the speech-recognition models (HuBERT, WavLM, wav2vec2) start trusting the <em>words</em> &mdash; but only in their <strong>deep</strong> layers. <strong>Whisper never does</strong>: it stays grounded in the voice at every layer. The music model (MERT) and w2v-BERT never cheat either.</p></div>
</section>

<h2 id="figs-h">Figures</h2>
<figure><img src="{F['summary']}" alt="overview">
<figcaption>The whole study at a glance (six panels): depth, acoustic-vs-lexical, cross-model overlap, features per model, cross-speaker, and the causal test.</figcaption></figure>
<figure><img src="{F['depth']}" alt="depth">
<figcaption>Depth: a probe reads emotion best deep (layer 10, blue), but the clean emotion features peak early (layer 7, red). Decoding is not the same as disentangling.</figcaption></figure>
<figure><img src="{F['acoustic']}" alt="acoustic vs lexical">
<figcaption>Each emotion feature scored on acoustic vs. lexical prediction. Nearly all sit on the acoustic side &mdash; they track the voice, not the words.</figcaption></figure>
<figure><img src="{F['emis']}" alt="EMIS by layer">
<figcaption>The "use-the-words" shortcut by layer. It appears only in the deep layers of the speech-recognition models; Whisper (red) stays voice-grounded at every layer.</figcaption></figure>
<figure><img src="{F['sufficiency']}" alt="sufficiency">
<figcaption>Causal test: emotion decodes well from acoustic features (0.55), poorly from lexical ones (0.43) &mdash; about the same as random (0.42).</figcaption></figure>

<section id="numbers">
<h2>5. The numbers at a glance</h2>
<table>
<tr><th>What</th><th>Result</th></tr>
<tr><td>How faithfully the SAE rebuilds the model's signal</td><td>93% of the variance</td></tr>
<tr><td>Genuine emotion features (after the confound check)</td><td>61 (from 394 candidates)</td></tr>
<tr><td>Emotion features that are acoustic, not lexical</td><td>60 of 61</td></tr>
<tr><td>Emotion organised earliest / read best at layer</td><td>L7 / L10</td></tr>
<tr><td>Decode emotion from acoustic / lexical / random features</td><td>0.55 / 0.43 / 0.42</td></tr>
<tr><td>Accuracy drop on speakers never seen in training</td><td>&approx; 0</td></tr>
<tr><td>Whisper "uses-the-words" score on mismatched speech</td><td class="no">&minus;0.11 (resists)</td></tr>
<tr><td>HuBERT / WavLM / wav2vec2 (deep layers)</td><td>+0.7 to +0.9 (use the words)</td></tr>
</table>
</section>

<section id="lit">
<h2>6. Recent literature &mdash; and how we're different</h2>
<p>We did a careful search of 2025&ndash;2026 work. The short version: our exact combination has <strong>not been done</strong>, but one very recent paper uses the same tool for a different purpose, so we position against it carefully.</p>
<table>
<tr><th>Paper</th><th>What it does</th><th>How we differ</th></tr>
<tr><td><strong>AudioSAE</strong> (EACL 2026)</td><td>Same SAE recipe on <em>two</em> models; even compares features across models.</td><td>Emotion is just one of four generic tasks &mdash; <span class="no">no</span> emotion-feature interpretation, confound check, fixed-sentence corpus, or acoustic-vs-lexical split. We do all of those, on <em>six</em> models. <strong>Closest neighbour.</strong></td></tr>
<tr><td>Pluth et al. (2026)</td><td>SAE on Whisper, but only the final layer; finds phonetic features.</td><td>No emotion target, no depth sweep.</td></tr>
<tr><td>Mariotte et al. (2026)</td><td>SAEs on audio models, same "beyond probing" framing.</td><td>Case study is <em>singing technique</em>, not emotion.</td></tr>
<tr><td>Ma et al. (2025)</td><td>Mechanistic study of Whisper for emotion.</td><td>Uses probing/CKA &mdash; <span class="no">no SAEs at all</span>.</td></tr>
</table>
<p><strong>Bottom line:</strong> our novel core &mdash; confound-controlled emotion features on a fixed-sentence corpus across six models, plus the depth and EMIS-by-depth findings &mdash; is, as far as we can tell, un-scooped. (Note: arXiv IDs should be double-checked before we cite them in a submission.)</p>
</section>

<section id="talk">
<h2>7. What to say in the presentation (per slide)</h2>
<p>Aim for ~10 minutes, one idea per slide. Here is a simple talk track anyone can deliver.</p>
<div class="pt"><strong>Slide 1 (Title):</strong> "We used sparse autoencoders to open up speech-emotion models and see which internal features carry emotion &mdash; and whether they listen to tone or words."</div>
<div class="pt"><strong>Slide 2 (The question):</strong> "These models top emotion benchmarks, but they were built to transcribe words. Are they hearing the <em>feeling</em> or reading the <em>words</em>? Probing can't tell us; we needed to look inside."</div>
<div class="pt"><strong>Slide 3 (Method):</strong> "An SAE is a fuse-box for the model. CREMA-D's fixed sentences are our control. The confound check is how we avoid fooling ourselves with phoneme detectors. 394 candidates become 61 real emotion features."</div>
<div class="pt"><strong>Slide 4 (Depth):</strong> "A probe reads emotion best deep (layer 10), but the clean features peak early (layer 7). Decoding is not the same as understanding."</div>
<div class="pt"><strong>Slide 5 (Across models &amp; speakers):</strong> "All six models share the same acoustic emotion features, and they work on people they've never heard. The features are about the emotion, not the person."</div>
<div class="pt"><strong>Slide 6 (Acoustic not lexical):</strong> "60 of 61 features track the voice, not the words. Emotion decodes from acoustic features (0.55) far better than lexical ones (0.43)."</div>
<div class="pt"><strong>Slide 7 (EMIS):</strong> "On mismatched speech, the speech-recognition models start trusting the words &mdash; but only deep down. Whisper never does. <em>Say the caveat out loud:</em> we use the raw activations here, not the SAE reconstruction, which flips Whisper by mistake."</div>
<div class="pt"><strong>Slide 8 (Literature):</strong> "One recent paper (AudioSAE) uses the same tool, but not for emotion and without our controls. Our affect-specific, controlled analysis is the new part."</div>
<div class="pt"><strong>Slide 9 (Limits &amp; future):</strong> "We rely on the sufficiency test, not ablation. Next: a real-transcript test on IEMOCAP, a cross-language study, and feature steering."</div>
<div class="pt"><strong>Slide 10 (Takeaways):</strong> "Decoding isn't understanding. The emotion these models use is in the <em>voice</em>, it's shared and speaker-independent, and only some models cheat with words &mdash; deep down."</div>
</section>

<section id="qa">
<h2>8. Likely questions (and answers)</h2>
<dl class="glossary">
<dt>Why CREMA-D and not IEMOCAP?</dt><dd>CREMA-D's fixed sentences are the control that lets us <em>separate</em> tone from words. IEMOCAP has real, varied transcripts &mdash; which is exactly why it's our planned next test.</dd>
<dt>Is 61 features too few?</dt><dd>It's after a deliberately strict triple filter (clear emotion link + single-meaning + beats both confounds). We want purity, not a big count; the looser set of 394 is mostly phonetics.</dd>
<dt>Does it depend on the SAE settings?</dt><dd>Reconstruction is stable (93%, almost no dead features) and the qualitative findings are robust. We haven't exhaustively swept every setting &mdash; that's listed as future work.</dd>
<dt>How is this different from AudioSAE?</dt><dd>Same tool, different question: they probe four generic attributes on two models; we do confound-controlled emotion-feature interpretation on six, plus the depth and EMIS-by-depth results.</dd>
<dt>Why does the EMIS reconstruction caveat matter?</dt><dd>If you run that one test on the SAE's rebuilt signal instead of the raw model signal, Whisper's result flips by mistake. The SAE is great for reading features, but not as a stand-in for the raw signal in that probe.</dd>
</dl>
</section>

<section id="limits">
<h2>9. Limitations &amp; what's next</h2>
<ul>
<li><strong>Sufficiency, not ablation.</strong> Emotion is spread across thousands of features, so removing a few barely hurts &mdash; we rely on the "can it decode from only these?" test.</li>
<li><strong>One layer per model</strong> for the cross-model and EMIS arms (not the full 25-layer grid for all six).</li>
<li><strong>Next experiments (already coded, waiting on data):</strong> a real-transcript causal test on IEMOCAP; a cross-language study (SpeechCraft) to explain why English&harr;Chinese transfer collapses; feature <em>steering</em> (turning an emotion feature up to change the prediction), which would make our claims causal rather than correlational.</li>
</ul>
</section>

<section id="where">
<h2>10. Where everything lives</h2>
<ul>
<li><strong>Code:</strong> <code>src/sae/</code> (library) and <code>src/sae/steps/</code> (the pipeline, one command per step).</li>
<li><strong>Results:</strong> <code>results/SAE/</code> (tables, plots, metrics) and its <code>summary/</code> folder.</li>
<li><strong>Write-ups:</strong> the 4-page report and slides in <code>presentation/</code>; the full technical brief in <code>presentation/SAE_branch_brief.md</code>; the literature search in <code>docs/sae/RELATED_WORK.md</code>.</li>
<li>Everything is on the <code>SAE</code> branch. This page is self-contained &mdash; it can be hosted directly on a GitHub page.</li>
</ul>
</section>

<footer>Prepared for the team to read and get presentation-ready. Questions &rarr; see the technical brief on the <code>SAE</code> branch.</footer>
</main>
</body>
</html>
"""

for out in [os.path.join(HERE, "SAE_briefing.html"),
            os.path.join(HERE, "..", "docs", "index.html")]:
    open(out, "w").write(HTML)
    print("wrote", os.path.relpath(out, os.path.join(HERE, "..")), f"({len(HTML)//1024} KB)")
