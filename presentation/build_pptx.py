#!/usr/bin/env python
"""Generate presentation/SAE_slides.pptx (mirrors the beamer deck, with author names)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "latex", "images")
NAVY = RGBColor(0x1F, 0x3A, 0x6E)
GREY = RGBColor(0x44, 0x44, 0x44)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation()
prs.slide_width = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def textbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tb.text_frame.word_wrap = True
    return tb.text_frame


def title_bar(slide, title):
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(0.95))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tf = bar.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.35); tf.margin_top = Inches(0.18)
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def bullets(slide, items, l=Inches(0.55), t=Inches(1.2), w=Inches(7.2), h=Inches(5.8), size=18):
    tf = textbox(slide, l, t, w, h)
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt; p.level = lvl
        p.font.size = Pt(size - 2 * lvl); p.font.color.rgb = GREY
        p.space_after = Pt(8)
    return tf


def pic(slide, path, l, t, w):
    if os.path.exists(path):
        slide.shapes.add_picture(path, l, t, width=w)


# 1 — title
s = prs.slides.add_slide(BLANK)
bar = s.shapes.add_shape(1, 0, Inches(1.9), SW, Inches(2.4))
bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
tf = bar.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.6)
p = tf.paragraphs[0]; p.text = "Sparse-Autoencoder Dissection of"
p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
p2 = tf.add_paragraph(); p2.text = "Emotion Features in Frozen Speech Encoders"
p2.font.size = Pt(34); p2.font.bold = True; p2.font.color.rgb = RGBColor(255, 255, 255)
p3 = tf.add_paragraph(); p3.text = "From decodability to mechanism"
p3.font.size = Pt(18); p3.font.italic = True; p3.font.color.rgb = RGBColor(0xCF, 0xDA, 0xF0)
auth = textbox(s, Inches(0.6), Inches(4.6), Inches(12), Inches(1.5))
p = auth.paragraphs[0]
p.text = "Nima Kelidari   ·   Minoo Ahmadi   ·   Chaitanya Parwatkar   ·   Xiangxu (Henry) Lin"
p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = NAVY
p = auth.add_paragraph(); p.text = "USC ICT — Intelligent Human Perception Lab"
p.font.size = Pt(15); p.font.color.rgb = GREY

# 2 — question
s = prs.slides.add_slide(BLANK); title_bar(s, "The question")
bullets(s, [
    ("Frozen speech encoders (HuBERT, WavLM, wav2vec2, w2v-BERT, MERT, Whisper) top SER benchmarks — but they were trained for ASR, so they encode a lot of TEXT.", 0),
    ("Probing tells us WHERE emotion is readable. It can't tell us:", 0),
    ("Which features encode emotion?", 1),
    ("Is the signal ACOUSTIC (how it's said) or LEXICAL (what's said)?", 1),
    ("Does the pattern hold across encoders and speakers?", 1),
    ("Idea: use Sparse Autoencoders to go from decodability → mechanism.", 0),
], w=Inches(12))

# 3 — method
s = prs.slides.add_slide(BLANK); title_bar(s, "Method in one slide")
bullets(s, [
    ("Train a TopK SAE (8×, k=32) on per-frame activations of each frozen encoder (~940k frames). Reconstruction: 93% variance, ~0 dead features.", 0),
    ("Anchor on CREMA-D: 91 actors × 12 FIXED sentences × 6 emotions → text is constant, so any emotion signal is acoustic (built-in control).", 0),
    ("Confound control: keep a feature only if its firing tells us more about EMOTION than about the SENTENCE (phonetics) or the SPEAKER.", 0),
    ("Result: 394 naive features → 61 genuine emotion features (median monosemanticity 0.59); dominated by anger & sadness.", 0),
    ("Without control, ~66% of ‘emotion’ features are really phonetic/speaker — the ‘SAE rediscovers phonemes’ trap.", 0),
], w=Inches(12))

# 4 — Q1 depth
s = prs.slides.add_slide(BLANK); title_bar(s, "Q1 — Depth: decoding ≠ disentangling")
bullets(s, [
    ("Probe accuracy peaks DEEP (L10) and stays high.", 0),
    ("But clean monosemantic emotion features peak EARLY (L7; monosemanticity L2).", 0),
    ("Deep layers keep decodability but lose disentanglement (24–47 feats).", 0),
    ("Takeaway: emotion is organised into clean features earlier than where a probe reads it best.", 0),
], w=Inches(6.4))
pic(s, os.path.join(IMG, "fig_depth.png"), Inches(7.1), Inches(1.4), Inches(5.9))

# 5 — Q2
s = prs.slides.add_slide(BLANK); title_bar(s, "Q2 — Across encoders & speakers")
bullets(s, [
    ("All six encoders carry emotion features (61–145 each), all anger/sadness-dominated.", 0),
    ("Cross-encoder overlap: Whisper→SSL 0.40 ≈ SSL→SSL 0.39 — on fixed-lexicon speech Whisper is NOT special; its difference is about TEXT.", 0),
    ("Speaker-invariant: leave-one-speaker-out (91 actors) drop ≈ 0; each feature fires for ~95% of its emotion's actors.", 0),
    ("→ the published MESD speaker-out collapse (~0.30) comes from the speaker-entangled features our control removes.", 0),
], w=Inches(12))

# 6 — Q3
s = prs.slides.add_slide(BLANK); title_bar(s, "Q3 — The signal is acoustic, not lexical")
bullets(s, [
    ("Positive control: 60/61 features label acoustic (eGeMAPS AUC 0.83 vs sentence 0.66).", 0),
    ("Causal sufficiency: emotion decodes at 0.55 from acoustic features vs 0.43 lexical vs 0.42 random (full 0.73).", 0),
    ("Takeaway: what the probe exploits is paralinguistic delivery, not words.", 0),
], w=Inches(6.4))
pic(s, os.path.join(IMG, "fig_sufficiency.png"), Inches(7.1), Inches(1.5), Inches(5.9))

# 7 — Q4 EMIS
s = prs.slides.add_slide(BLANK); title_bar(s, "Q4 — EMIS: the lexical shortcut is a deep-layer thing")
bullets(s, [
    ("EMIS = synthetic speech where TEXT emotion ≠ VOICE emotion.", 0),
    ("Train on congruent, test on incongruent; text-bias = text − audio accuracy.", 0),
    ("Shortcut appears ONLY in deep layers of ASR-SSL (HuBERT/WavLM/wav2vec2, +0.7–0.9).", 0),
    ("Whisper stays audio-grounded at EVERY layer (L24 −0.11); MERT/w2v-BERT never take it.", 0),
    ("Reproduces the published text-bias — and localizes it to depth.", 0),
], w=Inches(6.4))
pic(s, os.path.join(IMG, "fig_emis.png"), Inches(7.1), Inches(1.5), Inches(5.9))

# 8 — literature
s = prs.slides.add_slide(BLANK); title_bar(s, "Where this sits in the literature")
bullets(s, [
    ("AudioSAE (EACL 2026): same SAE recipe on 2 encoders, but emotion is just one probe task — NO emotion-feature interpretation, confound control, factorial corpus, or acoustic/lexical split. Closest method neighbor.", 0),
    ("ASR-SAE works (Pluth'26; Mariotte'26): SAEs on speech, but find phonetic/singing features — no affect, no depth sweep.", 0),
    ("Ma et al.'25: mechanistic Whisper-SER — but via probing/CKA, NO SAEs.", 0),
    ("Un-scooped core: confound-controlled SAE emotion features on a lexicon-controlled factorial corpus across 6 encoders + decode≠disentangle depth + EMIS-shortcut-by-depth.", 0),
], w=Inches(12))

# 9 — limitations + future
s = prs.slides.add_slide(BLANK); title_bar(s, "Limitations & future work")
bullets(s, [
    ("Limitations:", 0),
    ("Necessity-ablation is near-null (signal distributed) — we rely on sufficiency.", 1),
    ("SAE reconstruction must NOT replace raw activations for the EMIS probe (it inverted Whisper).", 1),
    ("Cross-encoder/EMIS arms use one layer per encoder, not full 25-layer grids.", 1),
    ("Future work:", 0),
    ("IEMOCAP real-transcript causal ablation (stub ready; needs transcripts).", 1),
    ("SpeechCraft cross-lingual feature overlap → explain EN↔ZH collapse.", 1),
    ("Feature steering; dimensional V/A/D targets; full per-encoder layer grids.", 1),
], w=Inches(12))

# 10 — takeaways
s = prs.slides.add_slide(BLANK); title_bar(s, "Takeaways")
bullets(s, [
    ("1. SAEs move SER analysis from decodability to mechanism; confound control is essential.", 0),
    ("2. Decoding ≠ disentangling: emotion organises early (L2–L7), decodes best deep (L10).", 0),
    ("3. Emotion features are acoustic and speaker-invariant, shared across all six encoders.", 0),
    ("4. The lexical shortcut is a deep-layer ASR-SSL phenomenon; Whisper resists it everywhere.", 0),
    ("All code (src/sae/), data pointers, artifacts (results/SAE/) and the report are on the SAE branch.", 0),
], w=Inches(12), size=19)

out = os.path.join(HERE, "SAE_slides.pptx")
prs.save(out)
print("wrote", out, "with", len(prs.slides._sldIdLst), "slides")
