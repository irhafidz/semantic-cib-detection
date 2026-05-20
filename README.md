**Semantic–Temporal Detection of Coordinated Inauthentic Behavior in Indonesian MBG Policy Discourse**
Code and annotation data for the paper submitted to **IEEE TENCON 2026**.

> I. Hafidz, C. Auriel, I. Hidayana, and N. A. Rakhmawati, “Adaptive Semantic Thresholding for CIB Detection Under Embedding Anisotropy: A Case Study of Indonesia’s MBG Program,” in Proc. IEEE TENCON, 2026


Irmasari Hafidz, Celine Auriel, and Nur Aini Rakhmawati, Department of Information Systems, Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia (e-mail: irma@its.ac.id; 5026221004@student.its.ac.id; nuraini@its.ac.id).

Irma Hidayana, Department of Population Health, School of Health Science, Hofstra University, New York, USA (e-mail: Irma.hidayana@hofstra.edu).


## 

| Path | Description |
|---|---|
| `iaa_fleiss_kappa.ipynb` | Inter-annotator agreement, Fleiss' κ for both annotation phases |
| `semantic_cib_detection.ipynb` | CIB detection pipeline on 96K tweets + T4 vs A100 benchmark |
| `human_ai_cib_validation.ipynb` | Grid search (72 configs), P/R/F1 evaluation, Figure 1 (anisotropy) |
| `dataset/` | Annotation CSVs (ground truth, N=945). Raw tweets included only for tweet_id, waktu (datetime), keyword, see inside folder. |

---

## Methodology
- We encode 96,584 Indonesian tweets about the MBG school-nutrition program with four Transformer encoders (MiniLM, IndoBERT, IndoBERTweet, XLM-RoBERTa). For each tweet, we collect all tweets posted within a forward temporal window W and compute cosine similarity against the anchor embedding.
- Tweets exceeding a model-specific threshold τ calibrated from the empirical similarity distribution at quantile q, are flagged as a CIB cluster. This **Adaptive Semantic Thresholding (AST)** corrects for embedding anisotropy, which causes fixed thresholds to behave inconsistently across architectures.
- Ground truth is 945 human-labeled tweets (Fleiss' κ = 0.92 after balanced sampling). IndoBERTweet achieves the best F1 = 0.56 (P = 0.50, R = 0.64) at q = 0.90, W = 600 s.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/irhafidz/semantic-cib-detection.git
cd semantic-cib-detection

# 2. Install dependencies
pip install sentence-transformers scikit-learn statsmodels scipy matplotlib seaborn pandas numpy torch

# 3. Run notebooks in order (Google Colab recommended — GPU required for step 2 and 3)
#    Step 1: iaa_fleiss_kappa.ipynb
#    Step 2: semantic_cib_detection.ipynb
#    Step 3: human_ai_cib_validation.ipynb
```

Set `PROJECT_FOLDER` in each notebook to your Google Drive path containing the data files.



## Reproducibility

- Python 3.10 · PyTorch 2.x · sentence-transformers 2.x
- Hardware: NVIDIA A100 40 GB (Google Colab Pro)
- All random operations use `RANDOM_SEED = 42`
- Threshold calibration uses a fixed 5,000-tweet sample drawn with `np.random.default_rng(42)`
- Full 96K corpus processes in ~95 s per model on A100; ~300 s on T4

---

## Key results

| Model | Quantile | Window | P | R | F1 |
|---|---|---|---|---|---|
| MiniLM | 0.90 | 600 s | 0.38 | 0.55 | 0.45 |
| IndoBERT | 0.90 | 600 s | 0.42 | 0.58 | 0.49 |
| **IndoBERTweet** | **0.90** | **600 s** | **0.50** | **0.64** | **0.56** |
| XLM-RoBERTa | 0.90 | 600 s | 0.40 | 0.72 | 0.51 |

Ground truth: N = 945 tweets · 5 annotators · Fleiss' κ = 0.92


## Data

The raw 96,584 tweets cannot be redistributed under Twitter/X API Terms of Service. The `dataset/` folder contains:

- `df-final-ground-truth.csv`: 945 annotated tweets (tweet_id, 5 annotator labels, majority-vote Ground_Truth)
- `positive-skewed-ground-truth.csv`: Phase 1 subset (N ≈ 450, κ = 0.41, documents the Kappa Paradox)
- 5 csv suspected buzzer list (anno_1.csv-anno_5.csv) and 5 csv suspected non-buzzer list (anno_nonbz_1.csv-anno_nonbz_5.csv) from 5 independent annotators

To reconstruct the full corpus, use the Twitter/X API v2 with the tweet IDs in the ground truth file and the keywords: `makan bergizi gratis`, `makan siang gratis`, `mbg`, `#mbg`, `keracunan mbg`, `manfaat mbg`.

## Citation
If you use this code or dataset, please cite: (if accepted)

```bibtex
@inproceedings{hafidz2026cib,
  title     = {Semantic--Temporal Detection of Coordinated Inauthentic Behavior
               in Indonesian MBG Policy Discourse},
  author    = {[Hafidz, I., Auriel, C., Hidayana, I., Rakhmawati, N.A.]},
  booktitle = {Proceedings of IEEE TENCON 2026},
  year      = {2026},
}
```

## License
MIT — see `LICENSE`.
