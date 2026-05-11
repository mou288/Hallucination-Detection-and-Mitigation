# Selective RAG with Hallucination Detection

A medical question-answering system that combines Retrieval-Augmented Generation (RAG) with a learned hallucination classifier to selectively apply retrieval only when it is likely to help. Built on top of Mistral-7B-Instruct and a FAISS/BM25 hybrid vector database of First Aid medical content.

---

## Overview

Large language models frequently hallucinate on medical questions. Naive RAG retrieval sometimes makes things worse by injecting irrelevant context. This project addresses both problems:

1. **Hallucination classifier** — three neural sub-models trained on internal LLM signals (token probabilities, attention patterns, FFN activations) detect whether a generated answer is likely hallucinated, before any retrieval occurs.
2. **Selective RAG** — retrieval is triggered only when the classifier flags the baseline answer as potentially hallucinated. This avoids degrading answers that are already correct.
3. **Abstention** — when the classifier is highly confident but retrieval still produces a low-quality answer, the system abstains rather than serving a wrong response.

### Final Results

| Metric | Value |
|---|---|
| Baseline similarity | 0.7966 |
| RAG similarity (selective) | 0.8144 |
| Improvement rate | 7.97% |
| Hallucination reduction | 19.82% |
| Coverage | 93.4% |
| Abstention rate | 6.6% |

Category breakdown across 5,001 evaluated questions:

| Category | Count |
|---|---|
| Already correct (no retrieval needed) | 3936 |
| Fixed by RAG | 126 |
| Safe abstain | 91 |
| Ruined by abstain | 241 |
| Still wrong | 540 |
| Broken by RAG | 67 |

---

## Architecture

```
Question
   │
   ▼
Mistral-7B generates baseline answer
   │
   ▼
Internal signals extracted
   ├── Softmax features (top-20 token probabilities)  [dim: 20]
   ├── Attention features (last-layer attention row)  [dim: 2048]
   └── FFN activation (last MLP gate × up projection) [dim: 14336]
   │
   ▼
Weighted Ensemble Classifier
   ├── SoftmaxClassifier  (weight: 0.20)
   ├── AttentionClassifier (weight: 0.30)
   └── FFNClassifier       (weight: 0.50)
   │
   ├── Not hallucinated → return baseline answer
   └── Hallucinated →
          │
          ▼
       Hybrid retrieval (FAISS dense + BM25 sparse)
       + Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
       + UMLS entity filtering
          │
          ▼
       Mistral-7B regenerates with retrieved context
          │
          ├── Confident → return RAG answer
          └── Still uncertain → abstain
```

---

## Repository Structure

```
├── vector_db.ipynb                          # Build the FAISS + BM25 hybrid vector database
├── Data_generation_for_classifier.ipynb     # Extract LLM internal signals and label training data
├── classifier_weighted_ensemble.py          # Train and evaluate the hallucination classifier
└──RAG_pipeline_generation_and_evaluation.ipynb  # End-to-end inference pipeline with evaluation

```

---

## Components

### Vector Database (`vector_db.ipynb`)

Builds a hybrid retrieval index over medical text (USMLE step 1, USMLE step 2, pathoma, pharmacology).

- **Chunking**: CUI-aware chunking using scispaCy + UMLS entity linking. Chunks are split on semantic dissimilarity and medical concept boundaries rather than fixed token windows.
- **Dense index**: FAISS with `all-mpnet-base-v2` embeddings (normalized, cosine similarity).
- **Sparse index**: BM25 (rank-bm25) over tokenized chunks.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for final passage reranking.

Artifacts produced: `faiss.index`, `bm25.pkl`, `metadata.pkl`, `chunks.json`.

### Data Generation (`Data_generation_for_classifier.ipynb`)

Generates a labeled dataset of (question, answer, hallucinated?) triples from `medalpaca/medical_meadow_medical_flashcards`.

For each question:
- Mistral-7B generates an answer.
- Labeling uses a combination of SBERT cosine similarity, NLI entailment (RoBERTa-large-MNLI), and semantic overlap against the ground truth answer.
- Internal LLM signals are extracted from two forward passes: softmax features from the logit distribution, attention row from the last transformer layer, and FFN activations from the last MLP block.

Output: a `.npz` file containing `X_softmax`, `X_attn_padded`, `X_ffn`, and `y` (15,177 samples; 32.1% hallucinated).

### Hallucination Classifier (`classifier_weighted_ensemble.py`)

Three lightweight neural classifiers trained on the extracted internal signals:

| Sub-model | Input dim | Architecture | Dropout | Val F1 |
|---|---|---|---|---|
| Softmax | 20 | 20 → 32 → 16 → 2 | 0.50 | 0.511 |
| Attention | 2048 | 2048 → 128 → 64 → 2 | 0.65 | 0.516 |
| FFN | 14336 | 14336 → 128 → 64 → 2 | 0.70 | 0.532 |

Training details:
- Loss: CrossEntropyLoss with balanced class weights
- Optimizer: Adam, weight decay 1e-3, gradient clipping 1.0
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: patience 8–10 per sub-model
- Calibration: isotonic regression per sub-model, fit on the validation set
- Ensemble: weighted average of calibrated probabilities (FFN=0.50, Attention=0.30, Softmax=0.20)
- Threshold: 0.3172 (optimized for F1 on validation set)

Ensemble test set results (at optimal threshold):

| Metric | FFN (best single) | Weighted Ensemble |
|---|---|---|
| F1 | 0.5615 | 0.5659 |
| ROC-AUC | 0.7132 | 0.7242 |
| Recall (hallucinated) | 0.6963 | 0.7223 |
| Precision (hallucinated) | 0.4704 | 0.4652 |

The ensemble improves over the best single model by +0.44 F1 and +1.10 ROC-AUC, and beats the random baseline by 16.4% relative F1.

Artifacts saved: `weighted_ensemble.pth`, `weighted_scalers.pkl`.

### RAG Pipeline (`RAG_pipeline_generation_and_evaluation.ipynb`)

End-to-end inference loop evaluated against a held-out set of medical Q&A pairs.

- Baseline answer generated with Mistral-7B (4-bit NF4 quantization via bitsandbytes).
- Classifier runs on the baseline answer's internal signals.
- If flagged as hallucinated, hybrid retrieval fetches top passages (FAISS + BM25 + reranker), filtered by UMLS entity overlap and a similarity threshold of 0.70.
- Mistral-7B regenerates the answer conditioned on the retrieved passages.
- An LLM judge (1,249 calls) scores final answers against ground truth for evaluation.

---

## Setup

This project was developed on Kaggle (data generation) and Google Colab (vector DB, RAG pipeline). GPU access is required for inference; the classifier training runs on CPU.

**Dependencies:**

```bash
pip install numpy==1.26.4 faiss-cpu scikit-learn scipy
pip install transformers accelerate bitsandbytes sentencepiece
pip install sentence-transformers tqdm joblib rank-bm25
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

**Models required (via Hugging Face):**
- `mistralai/Mistral-7B-Instruct-v0.3` 
- `sentence-transformers/all-mpnet-base-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `roberta-large-mnli`

**Running order:**

1. `vector_db.ipynb` — build the retrieval index
2. `Data_generation_for_classifier.ipynb` — generate and label training data
3. `classifier_weighted_ensemble.py` — train the ensemble
4. `RAG_pipeline_generation_and_evaluation.ipynb` — run the full pipeline

---

## Limitations

- The classifier was trained and evaluated on medical flashcard-style Q&A. Performance on other domains or longer-form answers is not characterized.
- The abstention mechanism reduces hallucinations but also introduces coverage loss (6.6% abstention rate).
- Retrieval is limited to the medical First Aid styled corpus. Questions outside this scope may not benefit from RAG.
- All inference uses 4-bit quantized Mistral-7B; results may differ with larger or unquantized models.

---

## License

See [LICENSE](LICENSE).
