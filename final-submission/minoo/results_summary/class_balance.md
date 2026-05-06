# Task 4 — Dataset class-balance audit

_Source: midterm report Appendix A Table 4. IEMOCAP and MSP-Podcast use the standard 4-class evaluation subsets (the same labels the LOSO probes were trained on)._

**Norm. entropy** = Shannon entropy / log(N_classes); 1.0 = perfectly balanced. **Imbalance ratio** = majority count / minority count; 1.0 = perfectly balanced.

| Dataset | # Cls | # Samp | Majority | Maj % | Minority | Min % | Baseline | Imb. Ratio | Norm. H | Label |
| :--- | ---: | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| RAVDESS | 8 | 2,880 | Calm | 13.3 | Neutral | 6.7 | 0.133 | 2.00 | 0.991 | Mostly balanced |
| EmoDB | 7 | 535 | Angry | 23.7 | Disgust | 8.6 | 0.237 | 2.76 | 0.978 | Imbalanced |
| IEMOCAP | 4 | 5,531 | Neutral | 30.9 | Sad | 19.6 | 0.309 | 1.58 | 0.984 | Mostly balanced |
| SAVEE | 7 | 480 | Neutral | 25.0 | Anger | 12.5 | 0.250 | 2.00 | 0.980 | Mostly balanced |
| AESDD | 5 | 604 | Anger | 20.0 | Sadness | 19.9 | 0.200 | 1.01 | 1.000 | Balanced |
| MESD | 6 | 862 | Anger | 16.7 | Neutral | 16.6 | 0.167 | 1.01 | 1.000 | Balanced |
| MSP-Podcast | 4 | 10,779 | Neutral | 50.7 | Sadness | 5.0 | 0.507 | 10.18 | 0.761 | Severely imbalanced |

## Interpretation

- **MSP-Podcast** is the only `Severely imbalanced` dataset: 50.7% of samples are neutral, imbalance ratio 10.2× (5469 neutral vs 537 sad). The majority-class baseline is already 0.507, so any probe scoring ~0.55–0.60 accuracy is barely beating 'always predict neutral'.
- The 6 balanced datasets all have normalized entropy ≥ 0.94 and imbalance ratio ≤ 2.6, so accuracy is a meaningful metric for them.
- **Recommendation:** for MSP-Podcast, report macro-UAR / macro-F1 instead of accuracy, and either class-weight the probe or stratify-downsample. This is documented as future work; in this version of the analysis, MSP-Podcast is reported separately (§10 of the report).