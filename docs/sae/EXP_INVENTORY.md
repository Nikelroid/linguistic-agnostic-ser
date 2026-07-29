# EXP embedding inventory

Root: `/scratch1/kelidari/ser-experiments`

## EXP4

- combos: 36 (clean: 36)
- encoders: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- datasets: ['AESDD', 'EmoDB', 'IEMOCAP', 'MESD', 'RAVDESS', 'SAVEE']
- example hidden_states shape: (604, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| HuBERT | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| HuBERT | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| HuBERT | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| HuBERT | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| HuBERT | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| MERT | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| MERT | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| MERT | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| MERT | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| MERT | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| MERT | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| WavLM | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| WavLM | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| WavLM | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| WavLM | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| WavLM | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| WavLM | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| Whisper | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| Whisper | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| Whisper | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| Whisper | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| Whisper | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| Whisper | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| w2v-BERT | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| w2v-BERT | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| w2v-BERT | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| w2v-BERT | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| w2v-BERT | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| w2v-BERT | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| wav2vec2 | AESDD | (604, 25, 1024) | clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| wav2vec2 | EmoDB | (535, 25, 1024) | clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| wav2vec2 | IEMOCAP | (5460, 25, 1024) | clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| wav2vec2 | MESD | (862, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| wav2vec2 | RAVDESS | (2316, 25, 1024) | clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| wav2vec2 | SAVEE | (480, 25, 1024) | clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |

## EXP5

- combos: 18 (clean: 18)
- encoders: ['HuBERT', 'WavLM', 'Whisper']
- datasets: ['AESDD', 'EmoDB', 'IEMOCAP', 'MESD', 'RAVDESS', 'SAVEE']
- example hidden_states shape: (604, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | AESDD | (604, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| HuBERT | EmoDB | (535, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| HuBERT | IEMOCAP | (5460, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| HuBERT | MESD | (862, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| HuBERT | RAVDESS | (2316, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| HuBERT | SAVEE | (480, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| WavLM | AESDD | (604, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| WavLM | EmoDB | (535, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| WavLM | IEMOCAP | (5460, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| WavLM | MESD | (862, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| WavLM | RAVDESS | (2316, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| WavLM | SAVEE | (480, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |
| Whisper | AESDD | (604, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,sadness | `d14 (1).wav` |
| Whisper | EmoDB | (535, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | bored,disgust,fearful,happy,neutral,sad | `12a02Wc.wav` |
| Whisper | IEMOCAP | (5460, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | ang,hap,neu,sad | `Ses01F_impro01_F000.wav` |
| Whisper | MESD | (862, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness | `Fear_F_A_fuera.wav` |
| Whisper | RAVDESS | (2316, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | angry,calm,disgust,fearful,happy,neutral,sad,surprised | `03-01-07-01-02-01-21.wav` |
| Whisper | SAVEE | (480, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,disgust,fear,happiness,neutral,sadness,surprise | `JE_sa09.wav` |

## EXP6

- combos: 6 (clean: 6)
- encoders: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- datasets: ['MSP-Podcast']
- example hidden_states shape: (13821, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast | (13821, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,contempt,disgust,fear,happiness,neutral,sadness,sur... | `MSP-PODCAST_0001_0008.wav` |

## EXP7

- combos: 6 (clean: 6)
- encoders: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- datasets: ['MSP-Podcast']
- example hidden_states shape: (10779, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast | (10779, 25, 1024) | SNR0,SNR10,SNR20,SNR5,clean | anger,happiness,neutral,sadness | `MSP-PODCAST_0001_0008.wav` |

## EXP8

- combos: 18 (clean: 18)
- encoders: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- datasets: ['MSP-Podcast_EmoAct', 'MSP-Podcast_EmoDom', 'MSP-Podcast_EmoVal']
- example hidden_states shape: (16903, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| HuBERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| HuBERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoAct | (16903, 25, 1024) | clean | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoDom | (16903, 25, 1024) | clean | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoVal | (16903, 25, 1024) | clean | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |

## EXP9

- no `embeddings/` dir; subdirs: ['.ipynb_checkpoints', 'summary']

## EXP10

- combos: 18 (clean: 0)
- encoders: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- datasets: ['MSP-Podcast_EmoAct', 'MSP-Podcast_EmoDom', 'MSP-Podcast_EmoVal']
- example hidden_states shape: (16903, 25, 1024) dtype=float32

| encoder | dataset | shape | SNR variants | labels | example filename |
|---|---|---|---|---|---|
| HuBERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| HuBERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| HuBERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| MERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| WavLM | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| Whisper | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| w2v-BERT | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoAct | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.333333,1.4,1.5,1.6,1.666667,1.8,1.809524,1.833333,1... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoDom | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.166667,1.2,1.333333,1.5,1.6,1.666667,1.727273,1.8,1... | `MSP-PODCAST_0001_0008.wav` |
| wav2vec2 | MSP-Podcast_EmoVal | (16903, 25, 1024) | SNR0,SNR10,SNR20,SNR5 | 1.0,1.052632,1.071429,1.083333,1.166667,1.2,1.230769,1.27... | `MSP-PODCAST_0001_0008.wav` |

## EXP20

- no `embeddings/` dir; subdirs: ['csv', 'summary']

## Summary

- all encoders seen: ['HuBERT', 'MERT', 'WavLM', 'Whisper', 'w2v-BERT', 'wav2vec2']
- all datasets seen: ['AESDD', 'EmoDB', 'IEMOCAP', 'MESD', 'MSP-Podcast', 'MSP-Podcast_EmoAct', 'MSP-Podcast_EmoDom', 'MSP-Podcast_EmoVal', 'RAVDESS', 'SAVEE']
