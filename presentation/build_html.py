#!/usr/bin/env python
"""Generate a single self-contained bilingual (EN/FA) HTML briefing for the SAE study.
Embeds key figures as base64 so the file is portable (drop on any GitHub page)."""
import base64, os

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "latex", "images")


def b64(name):
    p = os.path.join(IMG, name)
    if not os.path.exists(p):
        return ""
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


FIG_SUMMARY, FIG_DEPTH, FIG_EMIS, FIG_SUFF = (
    b64("fig_summary.png"), b64("fig_depth.png"), b64("fig_emis.png"), b64("fig_sufficiency.png"))

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAE Dissection of Speech Emotion Recognition — Team Briefing</title>
<style>
  :root {{ --navy:#1f3a6e; --red:#c0392b; --ink:#222; --soft:#f4f6fb; --line:#dde3ee; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,'Noto Sans Arabic',Tahoma,sans-serif;
         color:var(--ink); line-height:1.65; margin:0; background:#fff; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:24px 20px 80px; }}
  header {{ background:var(--navy); color:#fff; padding:26px 20px; }}
  header .wrap {{ padding-bottom:8px; padding-top:0; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ color:var(--navy); border-bottom:2px solid var(--line); padding-bottom:5px; margin-top:34px; }}
  h3 {{ color:var(--navy); margin-bottom:4px; }}
  .sub {{ opacity:.9; font-size:15px; }}
  .authors {{ font-size:13px; opacity:.85; margin-top:8px; }}
  .toggle {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line);
            padding:10px 0; text-align:center; }}
  .toggle button {{ font-size:15px; padding:7px 18px; margin:0 4px; border:1px solid var(--navy);
            background:#fff; color:var(--navy); border-radius:6px; cursor:pointer; }}
  .toggle button.active {{ background:var(--navy); color:#fff; }}
  .card {{ background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:14px 0; }}
  .finding {{ border-left:4px solid var(--navy); padding-left:14px; margin:14px 0; }}
  table {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:14px; }}
  th,td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; }}
  th {{ background:var(--navy); color:#fff; }}
  tr:nth-child(even) td {{ background:var(--soft); }}
  .resist {{ color:var(--red); font-weight:bold; }}
  figure {{ margin:18px 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; }}
  figcaption {{ font-size:13px; color:#555; margin-top:6px; }}
  .pt {{ background:#fffbe6; border:1px solid #f0e0a0; border-radius:8px; padding:6px 12px; margin:8px 0; }}
  .fa {{ display:none; }}
  [dir=rtl] .finding {{ border-left:none; border-right:4px solid var(--navy); padding-left:0; padding-right:14px; }}
  [dir=rtl] th, [dir=rtl] td {{ text-align:right; }}
  code {{ background:#eef1f7; padding:1px 5px; border-radius:4px; font-size:90%; }}
  footer {{ color:#666; font-size:13px; border-top:1px solid var(--line); margin-top:40px; padding-top:14px; }}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Opening up Speech-Emotion AI &mdash; what's inside the models</h1>
  <div class="sub en">A friendly briefing on our Sparse-Autoencoder study</div>
  <div class="sub fa">یک توضیح ساده دربارهٔ مطالعهٔ ما با Sparse Autoencoder</div>
  <div class="authors">Nima Kelidari &middot; Minoo Ahmadi &middot; Chaitanya Parwatkar &middot; Xiangxu (Henry) Lin</div>
</div></header>

<div class="toggle">
  <button id="btn-en" class="active" onclick="setLang('en')">English</button>
  <button id="btn-fa" onclick="setLang('fa')">فارسی</button>
</div>

<div class="wrap">

<!-- ===================== WHAT IS THIS ===================== -->
<div class="en">
<h2>What is this, in one minute?</h2>
<p>Modern AI models are very good at guessing the emotion in a voice clip. But they were originally built to <em>transcribe words</em>, so nobody quite knew <strong>what they actually listen to</strong>: the <em>tone of voice</em>, or the <em>words being said</em>? And <strong>which</strong> internal parts of the model do the emotion-detecting?</p>
<p>We used a tool called a <strong>Sparse Autoencoder (SAE)</strong> to crack the model open and read its internal "features" one by one. Think of it as turning a tangled ball of wires into a labelled fuse-box, so we can point at the few wires that carry "emotion."</p>
<div class="card">
<h3>The clever bit</h3>
<p>We test on <strong>CREMA-D</strong>, where 91 actors say the <em>same 12 sentences</em> in 6 emotions. Because the words never change, anything the model learns about emotion there must come from the <em>voice</em>, not the text. We also added a <strong>confound check</strong>: we only trust a feature as an "emotion feature" if it tells us more about the emotion than about the sentence or the speaker &mdash; otherwise the model is really just detecting <em>sounds of letters</em>, not feelings.</p>
</div>
</div>
<div class="fa">
<h2>این کار در یک دقیقه چیست؟</h2>
<p>مدل‌های هوش مصنوعیِ امروزی در حدسِ احساسِ یک کلیپ صوتی خیلی خوب‌اند. اما اینها در اصل برای <em>نوشتنِ کلمات</em> ساخته شده‌اند، پس کسی دقیق نمی‌دانست که این مدل‌ها واقعاً به <strong>چه چیزی گوش می‌دهند</strong>: به <em>لحنِ صدا</em> یا به <em>خودِ کلمات</em>؟ و <strong>کدام</strong> بخش‌های درونیِ مدل، کارِ تشخیصِ احساس را انجام می‌دهند؟</p>
<p>ما از ابزاری به نام <strong>Sparse Autoencoder (SAE)</strong> استفاده کردیم تا مدل را باز کنیم و «ویژگی‌های» درونی‌اش را یکی‌یکی بخوانیم. مثل این است که یک کلافِ درهم‌پیچیدهٔ سیم را به یک جعبه‌فیوزِ برچسب‌خورده تبدیل کنیم تا بتوانیم همان چند سیمی را که «احساس» را حمل می‌کنند نشان دهیم.</p>
<div class="card">
<h3>نکتهٔ هوشمندانه</h3>
<p>ما روی <strong>CREMA-D</strong> آزمایش می‌کنیم؛ جایی که ۹۱ بازیگر، <em>۱۲ جملهٔ یکسان</em> را با ۶ احساس می‌گویند. چون کلمات هیچ‌وقت عوض نمی‌شوند، هر چیزی که مدل آنجا دربارهٔ احساس یاد بگیرد لزوماً از <em>صدا</em> می‌آید، نه از متن. یک <strong>کنترلِ مزاحم</strong> هم اضافه کردیم: یک ویژگی را وقتی «ویژگیِ احساس» می‌دانیم که دربارهٔ احساس بیشتر بگوید تا دربارهٔ جمله یا گوینده &mdash; وگرنه مدل در واقع فقط <em>صدای حروف</em> را تشخیص می‌دهد، نه احساس را.</p>
</div>
</div>

<!-- ===================== WHAT WE FOUND ===================== -->
<div class="en">
<h2>What we found (four things)</h2>
<div class="finding"><h3>1. The model tidies emotion up early, but reads it best later</h3>
<p>The clean, single-meaning emotion features appear in <strong>early-middle layers (2&ndash;7)</strong>. Yet a simple classifier scores highest <strong>deeper (layer 10)</strong>. So "easy to read off" and "neatly organised" are two different things.</p></div>
<div class="finding"><h3>2. All six models share the same emotion features &mdash; and they ignore the speaker</h3>
<p>Every encoder lands on a similar set of acoustic emotion features. They also keep working on <strong>speakers never seen in training</strong> (almost no accuracy drop). The features are about the emotion, not the person.</p></div>
<div class="finding"><h3>3. It's the tone of voice, not the words</h3>
<p><strong>60 of 61</strong> emotion features are explained by acoustic properties (pitch, loudness, voice quality), not by which sentence was spoken. Emotion can be read off the acoustic features (0.55) far better than off the lexical ones (0.43).</p></div>
<div class="finding"><h3>4. When a model <em>can</em> cheat with words, only some do &mdash; and only deep down</h3>
<p>On a special "mismatched" test set (the written emotion differs from the spoken one), the self-supervised models start trusting the <em>words</em> &mdash; but only in their <strong>deep layers</strong>. <strong>Whisper never does</strong>: it stays grounded in the voice at every layer.</p></div>
</div>
<div class="fa">
<h2>چه چیزی فهمیدیم (چهار چیز)</h2>
<div class="finding"><h3>۱. مدل احساس را زود مرتب می‌کند، اما دیرتر بهتر می‌خواندش</h3>
<p>ویژگی‌های تمیز و تک‌معنای احساس در <strong>لایه‌های اولیه-میانی (۲ تا ۷)</strong> ظاهر می‌شوند. اما یک طبقه‌بندِ ساده در <strong>لایه‌های عمیق‌تر (لایهٔ ۱۰)</strong> بالاترین دقت را می‌گیرد. پس «راحت خوانده‌شدن» و «مرتب سازمان‌یافته‌بودن» دو چیز متفاوت‌اند.</p></div>
<div class="finding"><h3>۲. هر شش مدل ویژگی‌های احساسیِ مشترک دارند &mdash; و گوینده را نادیده می‌گیرند</h3>
<p>هر انکودر به مجموعهٔ مشابهی از ویژگی‌های آکوستیکِ احساس می‌رسد. این ویژگی‌ها روی <strong>گوینده‌هایی که در آموزش دیده نشده‌اند</strong> هم کار می‌کنند (تقریباً بدون افتِ دقت). ویژگی‌ها دربارهٔ احساس‌اند، نه دربارهٔ شخص.</p></div>
<div class="finding"><h3>۳. مهم لحنِ صداست، نه کلمات</h3>
<p><strong>۶۰ از ۶۱</strong> ویژگیِ احساس با خواصِ آکوستیک (زیر و بمی، بلندی، کیفیتِ صدا) توضیح داده می‌شوند، نه با اینکه چه جمله‌ای گفته شده. احساس را از روی ویژگی‌های آکوستیک (۰٫۵۵) خیلی بهتر می‌توان خواند تا از روی ویژگی‌های لغوی (۰٫۴۳).</p></div>
<div class="finding"><h3>۴. وقتی مدل <em>می‌تواند</em> با کلمات تقلب کند، فقط بعضی‌ها می‌کنند &mdash; آن هم فقط در عمق</h3>
<p>روی یک مجموعهٔ آزمایشیِ خاص و «ناهم‌خوان» (که احساسِ نوشته با احساسِ گفته فرق دارد)، مدل‌های self-supervised شروع می‌کنند به اعتماد به <em>کلمات</em> &mdash; اما فقط در <strong>لایه‌های عمیق</strong>. اما <strong>Whisper هرگز این کار را نمی‌کند</strong>: در هر لایه به صدا وفادار می‌ماند.</p></div>
</div>

<!-- ===================== TABLE (EN) ===================== -->
<div class="en">
<h2>The numbers at a glance</h2>
<table>
<tr><th>What</th><th>Result</th></tr>
<tr><td>Model faithfully reconstructed by the SAE</td><td>93% of the signal</td></tr>
<tr><td>Real emotion features (after the confound check)</td><td>61 (from 394 candidates)</td></tr>
<tr><td>Emotion features that are acoustic (not lexical)</td><td>60 of 61</td></tr>
<tr><td>Emotion organised earliest / read best at layer</td><td>L7 / L10</td></tr>
<tr><td>Decode emotion from acoustic / lexical / random features</td><td>0.55 / 0.43 / 0.42</td></tr>
<tr><td>Accuracy drop on unseen speakers</td><td>≈ 0</td></tr>
<tr><td>Whisper "uses-the-words" score on mismatched speech</td><td class="resist">−0.11 (resists)</td></tr>
<tr><td>HuBERT / WavLM / wav2vec2 (deep layers)</td><td>+0.7 to +0.9 (use the words)</td></tr>
</table>
</div>
<div class="fa">
<h2>اعداد در یک نگاه</h2>
<table>
<tr><th>چه چیزی</th><th>نتیجه</th></tr>
<tr><td>بازسازیِ مدل توسط SAE</td><td>۹۳٪ از سیگنال</td></tr>
<tr><td>ویژگی‌های واقعیِ احساس (بعد از کنترلِ مزاحم)</td><td>۶۱ (از ۳۹۴ کاندید)</td></tr>
<tr><td>ویژگی‌های احساسی که آکوستیک‌اند (نه لغوی)</td><td>۶۰ از ۶۱</td></tr>
<tr><td>سازمان‌یافتنِ زودهنگام / بهترین خوانش در لایهٔ</td><td>L7 / L10</td></tr>
<tr><td>خواندنِ احساس از ویژگی‌های آکوستیک / لغوی / تصادفی</td><td>۰٫۵۵ / ۰٫۴۳ / ۰٫۴۲</td></tr>
<tr><td>افتِ دقت روی گوینده‌های دیده‌نشده</td><td>≈ ۰</td></tr>
<tr><td>امتیازِ «استفاده از کلماتِ» Whisper روی گفتارِ ناهم‌خوان</td><td class="resist">−۰٫۱۱ (مقاومت می‌کند)</td></tr>
<tr><td>HuBERT / WavLM / wav2vec2 (لایه‌های عمیق)</td><td>۰٫۷+ تا ۰٫۹+ (از کلمات استفاده می‌کنند)</td></tr>
</table>
</div>

<!-- ===================== FIGURES ===================== -->
<figure><img src="{FIG_DEPTH}" alt="depth sweep">
<figcaption class="en">Depth: a probe reads emotion best deep (L10), but the clean features peak early (L7).</figcaption>
<figcaption class="fa">عمق: probe احساس را در عمق (L10) بهتر می‌خواند، اما ویژگی‌های تمیز زود (L7) به اوج می‌رسند.</figcaption></figure>

<figure><img src="{FIG_EMIS}" alt="EMIS text-bias by layer">
<figcaption class="en">The "use-the-words" shortcut appears only in the deep layers of the SSL models; Whisper (red) stays voice-grounded everywhere.</figcaption>
<figcaption class="fa">میان‌بُرِ «استفاده از کلمات» فقط در لایه‌های عمیقِ مدل‌های SSL ظاهر می‌شود؛ Whisper (قرمز) همه‌جا به صدا وفادار می‌ماند.</figcaption></figure>

<figure><img src="{FIG_SUMMARY}" alt="overview">
<figcaption class="en">The whole study at a glance (six panels): depth, acoustic-vs-lexical, cross-model overlap, features per model, cross-speaker, and the causal test.</figcaption>
<figcaption class="fa">کلِ مطالعه در یک نگاه (شش پنل): عمق، آکوستیک در برابر لغوی، همپوشانیِ بین‌مدلی، ویژگی‌ها به‌ازای هر مدل، بین‌گوینده، و آزمونِ علّی.</figcaption></figure>

<!-- ===================== TALK GUIDE ===================== -->
<div class="en">
<h2>What to say in the presentation</h2>
<p>Keep it to ~10 minutes. One idea per slide. Suggested flow:</p>
<div class="pt"><strong>Open with the hook:</strong> "These models ace emotion benchmarks &mdash; but are they hearing the <em>feeling</em>, or just reading the <em>words</em>? We opened them up to find out."</div>
<div class="pt"><strong>Explain the method simply:</strong> SAE = a fuse-box for the model. CREMA-D's fixed sentences = the control. The confound check = how we avoid fooling ourselves with phoneme detectors.</div>
<div class="pt"><strong>Walk the four findings</strong> (one slide each): early-vs-deep; shared &amp; speaker-invariant; acoustic-not-lexical; the deep-layer word-shortcut that only Whisper avoids.</div>
<div class="pt"><strong>Position honestly:</strong> one recent paper (AudioSAE) uses the same tool, but not for emotion and without our controls &mdash; our affect-specific, controlled analysis is the new part.</div>
<div class="pt"><strong>Close with the takeaway:</strong> "Decoding isn't the same as understanding. The emotion these models use is in the <em>voice</em>, it's shared and speaker-independent, and only some models cheat with words &mdash; deep down."</div>
<p><strong>Tips:</strong> say the EMIS caveat out loud (use raw activations, not the SAE reconstruction); have the key-numbers table ready for questions; and point people to the 4-page report and the code on the <code>SAE</code> branch.</p>
</div>
<div class="fa">
<h2>در ارائه چه بگوییم</h2>
<p>حدود ۱۰ دقیقه. هر اسلاید یک ایده. مسیرِ پیشنهادی:</p>
<div class="pt"><strong>با یک قلاب شروع کنید:</strong> «این مدل‌ها در بنچمارک‌های احساس عالی‌اند &mdash; اما آیا <em>احساس</em> را می‌شنوند یا فقط <em>کلمات</em> را می‌خوانند؟ ما بازشان کردیم تا بفهمیم.»</div>
<div class="pt"><strong>روش را ساده توضیح دهید:</strong> SAE = جعبه‌فیوزِ مدل. جملاتِ ثابتِ CREMA-D = کنترل. کنترلِ مزاحم = راهی که با آن خودمان را با تشخیص‌گرهای فُنِم گول نمی‌زنیم.</div>
<div class="pt"><strong>چهار یافته را مرور کنید</strong> (هر کدام یک اسلاید): زود در برابر عمیق؛ مشترک و مستقل از گوینده؛ آکوستیک نه لغوی؛ میان‌بُرِ لایه‌عمیقِ کلمات که فقط Whisper از آن دوری می‌کند.</div>
<div class="pt"><strong>صادقانه موقعیت‌یابی کنید:</strong> یک مقالهٔ اخیر (AudioSAE) از همین ابزار استفاده می‌کند، اما نه برای احساس و نه با کنترل‌های ما &mdash; تحلیلِ کنترل‌شده و مخصوصِ احساسِ ما، بخشِ تازه است.</div>
<div class="pt"><strong>با پیامِ نهایی ببندید:</strong> «خواندن با فهمیدن فرق دارد. احساسی که این مدل‌ها استفاده می‌کنند در <em>صدا</em>ست، مشترک و مستقل از گوینده است، و فقط بعضی مدل‌ها &mdash; آن هم در عمق &mdash; با کلمات تقلب می‌کنند.»</div>
<p><strong>نکته‌ها:</strong> نکتهٔ EMIS را با صدای بلند بگویید (از فعال‌سازیِ خام استفاده کنید، نه بازسازیِ SAE)؛ جدولِ اعداد را برای پرسش‌ها آماده داشته باشید؛ و افراد را به گزارشِ ۴ صفحه‌ای و کدِ روی برنچِ <code>SAE</code> ارجاع دهید.</p>
</div>

<footer>
<span class="en">All code, data pointers, figures, the 4-page report and the slides live on the <code>SAE</code> branch of the project repository. This page is self-contained &mdash; you can host it directly on a GitHub page.</span>
<span class="fa">همهٔ کد، اشاره‌گرهای داده، نمودارها، گزارشِ ۴ صفحه‌ای و اسلایدها روی برنچِ <code>SAE</code> در مخزن پروژه هستند. این صفحه مستقل است &mdash; می‌توانید مستقیماً روی یک صفحهٔ GitHub میزبانی‌اش کنید.</span>
</footer>
</div>

<script>
function setLang(l) {{
  document.querySelectorAll('.en').forEach(function(e){{ e.style.display = (l==='en')?'':'none'; }});
  document.querySelectorAll('.fa').forEach(function(e){{ e.style.display = (l==='fa')?'':'none'; }});
  document.body.dir = (l==='fa')?'rtl':'ltr';
  document.documentElement.lang = l;
  document.getElementById('btn-en').className = (l==='en')?'active':'';
  document.getElementById('btn-fa').className = (l==='fa')?'active':'';
}}
setLang('en');
</script>
</body>
</html>
"""

out = os.path.join(HERE, "SAE_briefing.html")
open(out, "w").write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
