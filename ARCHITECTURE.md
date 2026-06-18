# SecureMed-LLM — Architecture Documentation

> **Reference paper:** Boumezbeur et al., *"SecureMed-LLM: A Privacy-Preserving Framework
> for Safeguarding Clinical Language Models"*, PeerJ Computer Science (2025).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Six-Level Security Pipeline](#3-six-level-security-pipeline)
   - [Level 1 — Clinician Input](#level-1--clinician-input)
   - [Level 2 — Local Anonymization (Med-Guard)](#level-2--local-anonymization-med-guard)
   - [Level 3 — Secure Transmission](#level-3--secure-transmission)
   - [Level 4 — Privacy-Preserving Report Generation](#level-4--privacy-preserving-report-generation)
   - [Level 5 — Multi-Stage Clinical Validation (IDS-LLM)](#level-5--multi-stage-clinical-validation-ids-llm)
   - [Level 6 — Encrypted Delivery (ECIES/Curve25519)](#level-6--encrypted-delivery-eciescurve25519)
4. [Model Architecture](#4-model-architecture)
   - [BioMedCLIP Vision Encoder](#41-biomedclip-vision-encoder)
   - [T5 Text Decoder](#42-t5-text-decoder)
   - [SecureMedPipeline](#43-securemed-pipeline)
5. [Privacy-Preserving Training](#5-privacy-preserving-training)
   - [Differentially Private SGD (DP-SGD)](#51-differentially-private-sgd-dp-sgd)
   - [Adversarial Training](#52-adversarial-training)
6. [Anonymization Modules (Med-Guard)](#6-anonymization-modules-med-guard)
   - [Image Anonymization](#61-image-anonymization)
   - [Text Anonymization](#62-text-anonymization)
7. [Adversarial Attack Generation (BioMedAttack-LLM)](#7-adversarial-attack-generation-biomedattack-llm)
8. [Encryption Layer](#8-encryption-layer)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Training Hyperparameters](#10-training-hyperparameters)
11. [Dataset](#11-dataset)
12. [Inference Latency](#12-inference-latency)
13. [Key Results](#13-key-results)
14. [Limitations](#14-limitations)
15. [Module Dependency Graph](#15-module-dependency-graph)

---

## 1. System Overview

SecureMed-LLM is a **modular, multi-layered safeguarding framework** for
privacy-preserving clinical report generation from chest X-ray images.
It addresses three concurrent threat classes that prior work handles only in isolation:

| Threat class | Defense mechanism |
|---|---|
| Privacy leakage (re-identification via PHI) | Med-Guard text & image anonymization |
| Model memorization (membership inference) | DP-SGD training (ε=3.0, δ=10⁻⁵) |
| Adversarial manipulation (FGSM, PGD, prompt injection) | Adversarial fine-tuning (5% injection ratio) |
| Unsafe / hallucinated output | IDS-LLM three-stage post-generation validation |
| Unauthorized report access | ECIES/Curve25519 end-to-end encryption |

All five defenses are integrated into a **single, coordinated pipeline** rather than
applied independently, enabling layered protection across the full model lifecycle.

---

## 2. Repository Structure

```
securemed-llm/
├── src/
│   ├── anonymization/
│   │   ├── medguard_image.py       # Gaussian noise image obfuscation (Level 2)
│   │   └── medguard_text.py        # Presidio PHI de-identification (Level 2)
│   ├── models/
│   │   ├── biomed_clip_encoder.py  # BioMedCLIP ViT-B/16 wrapper
│   │   ├── t5_decoder.py           # T5-base + visual projection layer
│   │   └── pipeline.py             # SecureMedPipeline (encoder + decoder)
│   ├── privacy/
│   │   └── dp_training.py          # DP-SGD via Opacus (Level 4 training)
│   ├── adversarial/
│   │   ├── attack_generator.py     # BioMedAttack-LLM + image attack methods
│   │   └── adv_training.py         # Adversarial fine-tuning loop
│   ├── validation/
│   │   └── ids_llm.py              # IDS-LLM three-stage validator (Level 5)
│   ├── encryption/
│   │   ├── encrypt_report.py       # ECIES/Curve25519 encryption (Level 6)
│   │   └── decrypt_report.py       # ECIES/Curve25519 decryption (Level 6)
│   └── utils/
│       ├── data_loader.py          # Open-I dataset + DataLoader builders
│       └── metrics.py              # BLEU, SSIM, PSNR, MIA, PHI leakage
├── scripts/
│   ├── train.py                    # Offline DP-SGD + adversarial training
│   ├── inference.py                # End-to-end inference (all 6 levels)
│   └── evaluate.py                 # Full evaluation suite (paper metrics)
├── docs/
│   └── ARCHITECTURE.md             # This document
├── config/
│   └── config.yaml                 # Hyperparameters (Table 1)
└── keys/                           # Physician key storage (gitignored)
```

---

## 3. Six-Level Security Pipeline

The pipeline operates sequentially through six defence levels.
Encryption scope is **strictly limited to the final validated report**; model inputs,
intermediate embeddings, and inference-time activations are not encrypted (Section 5.1l).

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLINICIAN DEVICE (local — no raw PHI transmitted)                   │
│                                                                      │
│  Level 1: Raw chest X-ray + clinical notes                           │
│      │                                                               │
│      ▼                                                               │
│  Level 2: Med-Guard Anonymization                                    │
│    ├─ Image: Gaussian noise N(0, σ²), σ=15  [medguard_image.py]     │
│    └─ Text:  Presidio PHI redaction          [medguard_text.py]      │
│      │                                                               │
└──────┼───────────────────────────────────────────────────────────────┘
       │  TLS 1.3 (forward secrecy)
┌──────┼───────────────────────────────────────────────────────────────┐
│  SECURE SERVER (trusted, isolated environment)                       │
│                                                                      │
│  Level 3: Secure Transmission (HTTPS / TLS 1.3)                     │
│      │                                                               │
│      ▼                                                               │
│  Level 4: Privacy-Preserving Report Generation                       │
│    ├─ BioMedCLIP encoder → 512-d L2-normalised embedding            │
│    ├─ Linear projection → T5 embedding space                        │
│    ├─ T5-base decoder (beam search, n_beams=4)                      │
│    └─ Trained with DP-SGD (ε=3.0) + adversarial augmentation (5%)  │
│      │                                                               │
│      ▼                                                               │
│  Level 5: IDS-LLM Validation                                         │
│    ├─ Stage 1: Rule-based consistency (47 rules)                    │
│    ├─ Stage 2: Clinical entity verification (SNOMED-CT)             │
│    └─ Stage 3: Semantic anomaly detection (SBERT + Isolation Forest)│
│      │    [if rejected → regeneration loop, max 3 attempts]         │
│      ▼                                                               │
│  Level 6: ECIES/Curve25519 Encryption                                │
│    └─ Encrypt validated report with physician's public key          │
│      │                                                               │
└──────┼───────────────────────────────────────────────────────────────┘
       │  TLS 1.3
┌──────┼───────────────────────────────────────────────────────────────┐
│  CLINICIAN DEVICE (local decryption)                                 │
│                                                                      │
│    └─ Decrypt with physician's private key → plaintext report       │
└──────────────────────────────────────────────────────────────────────┘
```

### Level 1 — Clinician Input

Raw multimodal medical data is provided by the clinician:
chest X-ray images (`.png`/`.jpg`, any resolution) and optional clinical notes (`.txt`).
No processing occurs at this level.

### Level 2 — Local Anonymization (Med-Guard)

All operations run **locally on the clinician's device** before any data
leaves the local environment, preventing transmission of raw PHI.

**Image anonymization** (`src/anonymization/medguard_image.py`):
- Additive Gaussian noise N(0, σ²) applied to raw pixel values (uint8 scale).
- Operational setting: **σ = 15** (empirically selected on validation set).
- Output clipped to [0, 255] and returned as PIL Image for downstream compatibility.
- *Caveat (paper Section 5.1a):* Gaussian noise does **not** provide formal
  re-identification guarantees equivalent to differential privacy or k-anonymity.
  Its role is computational-lightweight visual obfuscation specific to this dataset.

**Text anonymization** (`src/anonymization/medguard_text.py`):
- Microsoft Presidio `AnalyzerEngine` + `AnonymizerEngine` with domain-specific
  medical NER extensions (`en_core_med7_lg` if installed).
- PHI entity types: `PERSON`, `DATE_TIME`, `LOCATION`, `MEDICAL_LICENSE`,
  `PHONE_NUMBER`, `EMAIL_ADDRESS`, `US_SSN`, `US_DRIVER_LICENSE`, `US_PASSPORT`,
  `URL`, `IP_ADDRESS`, `NRP`.
- Replaced with typed placeholders: `[NAME]`, `[DATE]`, `[TIME]`, `[LOCATION]`, etc.
- A rule-based time pre-pass handles patterns like `9:04 a.m.` → `[TIME]`.
- Confidence threshold: **0.35** (Presidio default, tuned for recall over precision).

### Level 3 — Secure Transmission

Anonymized data is transmitted to the secure processing server over **HTTPS / TLS 1.3**
with forward secrecy, providing confidentiality, integrity, and authentication.
This level is an infrastructure concern and is not implemented as a Python module.

### Level 4 — Privacy-Preserving Report Generation

The fine-tuned BioMedCLIP+T5 model generates a preliminary clinical report
from the anonymized image. See [Section 4](#4-model-architecture) for the
full architecture description.

Training defenses applied at this level:

- **DP-SGD** (Opacus): per-sample gradient clipping (C=1.0) + calibrated
  Gaussian noise (σ=1.1) → budget (ε=3.0, δ=10⁻⁵) after 5 epochs.
- **Adversarial augmentation**: 5% of each training batch replaced with
  adversarially perturbed samples (FGSM/PGD/DeepFool on images;
  BioMedAttack-LLM on text prompts).

### Level 5 — Multi-Stage Clinical Validation (IDS-LLM)

Three sequential post-generation checks are applied by `src/validation/ids_llm.py`.
Reports failing **any** stage trigger a **controlled regeneration** request
(up to 3 attempts in `scripts/inference.py`).

| Stage | Method | Paper F1 | Paper AUC |
|---|---|---|---|
| 1. Rule-based consistency | 47 hand-crafted rules (chest X-ray conventions) | 93.7% | 0.93 |
| 2. Clinical entity verification | SNOMED-CT radiology subset mapping | 91.5% | 0.91 |
| 3. Semantic anomaly detection | SBERT + Isolation Forest (contamination=0.1) | 89.0% | 0.90 |
| **Overall** | All three stages combined | **91.3%** | **0.94** |

*Limitation (paper Section 9):* Rules were authored by the research team with reference to
published guidelines; they have not been validated by board-certified radiologists.

### Level 6 — Encrypted Delivery (ECIES/Curve25519)

The validated report text is encrypted using ECIES (Elliptic Curve Integrated
Encryption Scheme) with Curve25519 key exchange.

**Key management:**
- Each physician holds an asymmetric X25519 key pair.
- The **private key** is stored locally on the clinician's device (PEM format, mode 0600).
- The **public key** is registered with the trusted server (32-byte raw format).

**Payload format** (60 B header + N B ciphertext):
```
[32 B] Ephemeral X25519 public key
[12 B] AES-GCM nonce
[16 B] AES-GCM authentication tag
[ N B] AES-256-GCM ciphertext of UTF-8 report bytes
```

**Cryptographic construction:**
1. Generate ephemeral X25519 key pair (per-message forward secrecy).
2. X25519 ECDH between ephemeral private key and recipient's public key → shared secret.
3. HKDF-SHA256 (info = `b"SecureMed-LLM-ECIES-v1"`) → 256-bit AES key.
4. AES-256-GCM encryption with random 96-bit nonce → ciphertext + 128-bit auth tag.
5. Pack header + ciphertext → self-contained binary payload.

*Scope limitation (paper Section 5.1l):* Encryption covers **only** the final validated
report. Model inputs, intermediate embeddings, and inference-time computations are **not**
encrypted. The inference server is assumed to be a trusted, isolated environment.
This is an explicitly acknowledged design trade-off; TEE-based solutions are future work.

---

## 4. Model Architecture

### 4.1 BioMedCLIP Vision Encoder

**Module:** `src/models/biomed_clip_encoder.py` → `BioMedCLIPEncoder`

**Model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
loaded via `open_clip` from HuggingFace Hub.

**Backbone:** ViT-B/16 pretrained on 15 million biomedical image–text pairs
(PMC scientific figures). Pre-training on domain data provides:
- Clinically meaningful joint image–text representations.
- Reduced distributional shift vs. general-purpose vision encoders on chest X-rays.
- Dense embeddings compatible with T5's transformer decoder.

**Output:** L2-normalised embedding of shape `(B, 512)`.
L2 normalisation is applied after the ViT forward pass to be consistent with the
CLIP contrastive training objective.

**Freezing policy (default):** The ViT backbone is **frozen** during fine-tuning.
Only the visual projection layer in `T5Decoder` is trained by default, reducing the
number of trainable parameters and limiting the per-sample gradient sensitivity
required by Opacus for DP-SGD compatibility.

```
Input: (B, 3, 224, 224) — ImageNet-normalised chest X-ray
    │
    ▼
ViT-B/16 backbone (open_clip) [FROZEN by default]
    │
    ▼
L2 normalise
    │
    ▼
Output: (B, 512) — L2-normalised visual embedding
```

### 4.2 T5 Text Decoder

**Module:** `src/models/t5_decoder.py` → `T5Decoder`

**Model:** `t5-base` from HuggingFace Transformers (`d_model = 512`).

**Visual prefix injection (Section 5.1c):**
A learnable **linear projection layer** (`nn.Linear(512, 1 × 512)`) maps the
BioMedCLIP embedding into one visual prefix token in the T5 embedding space.
The prefix token is prepended to the tokenised report embeddings before the
T5 encoder, conditioning generation on visual features without modifying the
T5 architecture:

```
BioMedCLIP embedding (B, 512)
    │
    ▼
Linear projection: 512 → 1 × 512  [TRAINABLE — Xavier init]
    │
    ▼
Visual prefix: (B, 1, 512)
    │
Concatenate with text token embeddings: (B, seq_len, 512)
    │
    ▼
T5 encoder → T5 decoder (beam search, n_beams=4)
    │
    ▼
Generated token ids → decoded report string
```

**Training objective:** Teacher-forcing cross-entropy loss over target report tokens.

**Why T5 and not a larger LM (Section 5.1c):** T5 was selected for its
controllability, offline deployability, and compatibility with Opacus DP-SGD,
which requires per-sample gradient computation. Larger autoregressive models
(LLaMA, GPT variants) incur substantially higher cost under DP-SGD and were not
evaluated due to hardware constraints.

### 4.3 SecureMed Pipeline

**Module:** `src/models/pipeline.py` → `SecureMedPipeline`

Integrates the encoder and decoder into a single `nn.Module` that serves as
the primary object passed to all training, inference, and evaluation components.

**Key methods:**

| Method | Purpose |
|---|---|
| `forward(images, input_ids, attention_mask, labels)` | Training forward pass; returns `loss` + `logits` |
| `generate_report(images)` | Inference: returns list of decoded report strings |
| `compute_log_prob(images, input_ids, attention_mask)` | Per-token log-prob for MIA scoring |
| `compute_cosine_similarity(images, input_ids, attention_mask)` | Image–text cosine sim (Section 5.1d) |
| `save_checkpoint(output_dir, epoch)` | Persist encoder + decoder state dicts |
| `load_checkpoint(path, device)` | Class method; restores from saved checkpoint |
| `trainable_parameters()` | Returns only `requires_grad=True` parameters |
| `projection_parameters()` | Visual projection layer parameters only |

---

## 5. Privacy-Preserving Training

### 5.1 Differentially Private SGD (DP-SGD)

**Module:** `src/privacy/dp_training.py` → `DPTrainer`, `train_with_dp`

**Library:** Opacus v1.x

**Mechanism (Section 5.1i):**
DP-SGD clips per-sample gradients to norm C=1.0 and adds calibrated Gaussian
noise with multiplier σ=1.1 at each step. After 5 epochs, this yields a
privacy budget of **(ε=3.0, δ=10⁻⁵)** under the **Rényi DP accountant**.

**Privacy budget rationale (Section 5.1h):**
ε=3.0 represents a moderate privacy guarantee — an empirically practical
operating point that limits MIA success to ~55% while maintaining BLEU-4=0.70.
It is **not** a strong DP guarantee by the standards of the DP literature.

**Opacus compatibility:** All BatchNorm layers are replaced with GroupNorm
(`ModuleValidator.fix()`) before attaching the PrivacyEngine, as required by Opacus.
The Opacus `GradSampleModule` wrapper is transparently unwrapped for checkpointing.

**Training configuration (Table 1):**

| Parameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay=0.01) |
| Learning rate | 2 × 10⁻⁵ |
| Batch size | 16 |
| Epochs | 5 |
| DP clipping norm C | 1.0 |
| DP noise multiplier σ | 1.1 |
| Privacy budget ε | 3.0 |
| Privacy budget δ | 10⁻⁵ |
| DP accountant | Rényi |

**Effect on MIA (Table 4):** DP-SGD reduces MIA accuracy from 85% (unprotected)
to **55%** at ε=3.0 — a 30 percentage-point absolute reduction approaching the
50% random-classifier baseline.

### 5.2 Adversarial Training

**Module:** `src/adversarial/adv_training.py` → `AdversarialTrainer`

**Injection ratio:** 5% of each training batch is replaced with adversarially
perturbed samples (selected by sweeping {3%, 5%, 7%, 10%} on the validation set;
Section 5.1g).

**Training flow:**
1. Load DP-SGD checkpoint (adversarial training builds on the privacy-trained model).
2. For each batch, randomly select 5% of samples.
3. Apply one of {FGSM, PGD, DeepFool} to the image, or substitute a
   pre-generated text adversarial report from BioMedAttack-LLM.
4. Forward the mixed clean/adversarial batch, compute cross-entropy loss.
5. Backpropagate with standard gradient clipping (C=1.0, no DP noise).
6. Save best checkpoint keyed on clean validation BLEU-4.

**Effect (Tables 2–3):**

| Attack | BLEU-4 (pre-fine-tuning) | BLEU-4 (post-fine-tuning) |
|---|---|---|
| Clean | — | Reference |
| FGSM (ε=0.1) | 0.45 | **0.68** |
| PGD (ε=0.1) | 0.33 | **0.63** |
| DeepFool | 0.29 | **0.51** |

---

## 6. Anonymization Modules (Med-Guard)

### 6.1 Image Anonymization

**Module:** `src/anonymization/medguard_image.py` → `MedGuardImageAnonymizer`

Applies additive Gaussian noise N(0, σ²) to raw PIL Image pixel values (uint8 scale),
clips to [0, 255], and returns a noised PIL Image for downstream BioMedCLIP preprocessing.

**Noise level selection (Table S6 / Section 5.1b):**

| σ | SSIM | BLEU-4 | PHI Leakage |
|---|---|---|---|
| 5 | ≈0.96 | 0.82 | ~15% (insufficient) |
| **15** | **≈0.81** | **0.70** | **2.1% (SELECTED)** |
| 25 | ≈0.65 | 0.58 | ~6% |
| 50 | ≈0.41 | 0.34 | ~3% (severe quality loss) |

**Important caveat:** Gaussian noise injection does **not** provide formal
re-identification guarantees equivalent to differential privacy or k-anonymity.
Its role is to reduce visual identifiability in a computationally lightweight manner.

### 6.2 Text Anonymization

**Module:** `src/anonymization/medguard_text.py` → `MedGuardTextAnonymizer`

**PHI entity → placeholder mapping:**

| Entity type | Placeholder |
|---|---|
| `PERSON` | `[NAME]` |
| `DATE_TIME` | `[DATE]` |
| `LOCATION` | `[LOCATION]` |
| `MEDICAL_LICENSE` | `[ID]` |
| `PHONE_NUMBER` | `[PHONE]` |
| `EMAIL_ADDRESS` | `[EMAIL]` |
| `US_SSN` / `US_DRIVER_LICENSE` / `US_PASSPORT` | `[ID]` |
| time patterns (rule-based pre-pass) | `[TIME]` |

**PHI leakage metric (Section 5.1a):** Proportion of source PHI entity values
(detected by Presidio on the original report) that appear verbatim in the
generated report. This is an approximation of re-identification risk, **not**
a formal privacy guarantee.

---

## 7. Adversarial Attack Generation (BioMedAttack-LLM)

**Module:** `src/adversarial/attack_generator.py`
→ `BioMedAttackLLM`, `ImageAttackGenerator`

**Text attacks — BioMedAttack-LLM (Section 5.1e):**
- Base model: `microsoft/phi-2`, fine-tuned on the Open-I training split.
- Three structured transformations per sample:
  1. Negation of clinical findings
  2. Omission of critical observations
  3. Insertion of misleading conclusions
- Each original report produces 2–3 adversarial variants.
- Hyperparameters: lr=2×10⁻⁵, batch_size=8, grad_accum=2, epochs=3,
  max_seq_length=512, linear scheduler with 10% warm-up.

**Image attacks — ImageAttackGenerator (Section 5.1g):**
- Attacks: **FGSM**, **PGD**, **DeepFool** via TorchAttacks library.
- Applied to a **ResNet18 surrogate** model (transfer attacks to BioMedCLIP,
  since BioMedCLIP is not differentiable via the attack library directly).
- Perturbation budgets: ε ∈ {0.1, 0.3, 0.5, 1.0} (L∞ norm).

**Acknowledged limitation (Section 9):** Attacks are transferred from a ResNet18
surrogate. The closed-world evaluation (same training distribution for attacker and
defender) may overestimate robustness against diverse real-world adversaries.
Stronger adaptive attacks (AutoAttack, Carlini–Wagner) are not evaluated.

---

## 8. Encryption Layer

**Modules:** `src/encryption/encrypt_report.py`, `src/encryption/decrypt_report.py`

**Cryptographic construction (ECIES — paper reference [38]):**

```
Encryption:
  1. Generate ephemeral X25519 key pair (per-message)
  2. X25519 ECDH(ephemeral_priv, recipient_pub) → 32-byte shared secret
  3. HKDF-SHA256(shared_secret, info="SecureMed-LLM-ECIES-v1") → 32-byte AES key
  4. AES-256-GCM(aes_key, nonce=random_96bit, plaintext=report_utf8) → (ciphertext, tag)
  5. Payload = ephemeral_pub[32] ‖ nonce[12] ‖ tag[16] ‖ ciphertext[N]

Decryption:
  1. Parse payload: ephemeral_pub[0:32], nonce[32:44], tag[44:60], ciphertext[60:]
  2. X25519 ECDH(recipient_priv, ephemeral_pub) → shared secret
  3. HKDF-SHA256(shared_secret, info="SecureMed-LLM-ECIES-v1") → AES key
  4. AES-256-GCM.decrypt(aes_key, nonce, ciphertext ‖ tag) → plaintext
  5. Return plaintext.decode("utf-8")
```

**Key management CLI:**
```bash
# Generate physician key pair
python -m src.encryption.encrypt_report keygen \
    --output_dir keys/ --physician_id dr_smith

# Encrypt a validated report
python -m src.encryption.encrypt_report encrypt \
    --report outputs/report.txt \
    --public_key keys/dr_smith_public.bin \
    --output outputs/report.enc

# Decrypt (on the clinician's device)
python -m src.encryption.decrypt_report \
    --payload outputs/report.enc \
    --private_key keys/dr_smith_private.pem \
    --output outputs/report_decrypted.txt
```

**Dependency:** `cryptography` (Python standard crypto library — no third-party ECIES
package required, improving auditability).

---

## 9. Evaluation Metrics

**Module:** `src/utils/metrics.py`

| Metric | Function | Purpose |
|---|---|---|
| BLEU-1, BLEU-4 | `compute_bleu()` | Text quality proxy (sacrebleu, corpus-level) |
| SSIM | `compute_ssim()` | Image structural similarity after anonymization |
| PSNR | `compute_psnr()` | Image quality in dB after anonymization |
| MIA accuracy | `compute_mia_accuracy()` | Black-box confidence-thresholding privacy attack |
| PHI leakage rate | `compute_phi_leakage()` | Presidio-based re-identification approximation |
| Cosine similarity | `cosine_similarity()` | Image–text semantic alignment (Section 5.1d) |

**Evaluation script** (`scripts/evaluate.py`) reproduces all paper tables:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/adv/best_checkpoint.pt \
    --test_dir  data/open-i/test \
    --train_dir data/open-i/train \
    --output_dir outputs/eval
```

Additional flags: `--noise_sweep` (Table S6), `--skip_mia`, `--skip_phi`, etc.

---

## 10. Training Hyperparameters

Full hyperparameter table from the paper (Table 1):

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Batch size | 16 |
| Epochs | 5 |
| Gradient clipping norm (standard) | 1.0 |
| DP-SGD clipping norm C | 1.0 |
| DP-SGD noise multiplier σ | 1.1 |
| Privacy budget ε | 3.0 |
| Privacy budget δ | 10⁻⁵ |
| DP accountant | Rényi |
| Max sequence length (T5) | 128 tokens |
| Image resolution | 224 × 224 pixels |
| ImageNet normalisation | mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) |
| Adversarial injection ratio | 5% |
| Med-Guard Gaussian noise σ | 15 |
| T5 beam search width | 4 |

---

## 11. Dataset

**Open-I chest X-ray dataset** — publicly available at:
[https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset](https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset)

This curated version of the NIH Open-I corpus provides paired chest X-ray images
and structured radiology reports (Findings + Impression sections).

**Splits (original, unmodified):**

| Split | Pairs |
|---|---|
| Training | 93,347 |
| Validation | 1,885 |
| Test | 1,541 |

**Expected directory layout:**
```
data/open-i/
    train/
        images/   (*.png or *.jpg)
        reports/  (*.txt — one per image, same stem)
    val/
        images/
        reports/
    test/
        images/
        reports/
```

**MIA evaluation note (Section 4):** The MIA member set is constructed from
**original** (un-anonymized) training samples only. The 3,682-pair count cited
in the paper refers to the original de-duplicated training pairs used for
privacy evaluation.

**Single-dataset limitation (Section 9):** All results are specific to this
dataset, originating from a single institution (Indiana University).
Generalization to other imaging modalities, institutions, or patient populations
cannot be assumed.

---

## 12. Inference Latency

Measured on CPU hardware (Intel Xeon, 2.0 GHz, 12 GB RAM) over 100 test reports
(Section 7.1):

| Component | Latency |
|---|---|
| Vision encoding (BioMedCLIP) | 0.8–2.5 s |
| Text generation (T5 decoder) | 2.5–5.5 s |
| IDS-LLM validation | 1.0–3.0 s |
| Projection / embedding fusion | < 0.05 s |
| **End-to-end (per report)** | **6.5 ± 2.3 s** |

Range: approximately 4.0–11.0 s depending on report length and image complexity.

SecureMed-LLM is **suitable** for near-real-time clinical decision support in
non-urgent radiology workflows. It is **not** suitable for real-time emergency triage.

---

## 13. Key Results

**Med-Guard ablation (Table 7):**

| Configuration | BLEU-4 | PHI Leakage |
|---|---|---|
| No Med-Guard (baseline) | 0.86 ± 0.01 | 18.7% |
| Text-only (Presidio) | 0.81 ± 0.02 | 6.3% |
| Image-only (σ=15) | 0.74 ± 0.02 | 14.9% |
| **Full Med-Guard** | **0.70 ± 0.01** | **2.1%** |

**Security component ablation (Table 8):**

| Configuration | BLEU-4 | MIA Acc. | Unsafe Output Rate |
|---|---|---|---|
| Full system | 0.70 | **55%** | **4.2%** |
| w/o DP-SGD | 0.74 | 78% | 5.1% |
| w/o Adversarial training | 0.73 | 61% | 12.8% |
| w/o IDS-LLM | 0.71 | 56% | 21.5% |

**Differential privacy vs. MIA (Table 4):**

| ε | MIA (%) | Utility (%) |
|---|---|---|
| 1.0 | 51.0 | 68.4 |
| 2.0 | 54.2 | 73.6 |
| **3.0** | **55.0** | **81.1 (selected)** |
| 4.0 | 59.4 | 83.0 |
| 5.0 | 64.5 | 85.6 |

---

## 14. Limitations

The following limitations are explicitly acknowledged in the paper (Section 9)
and are documented here for reproducibility:

- **Single-dataset evaluation.** All results are from the Open-I corpus (Indiana
  University). Generalization to CT, MRI, other pathologies, or multi-institutional
  data has not been demonstrated.
- **Weak MIA attack.** The black-box confidence-thresholding MIA is a weak attack.
  The reported 55% accuracy is an **upper bound on privacy protection**, not a lower
  bound. Shadow model and likelihood ratio attacks may achieve higher accuracy.
- **No white-box adversarial evaluation.** Robustness claims are limited to FGSM,
  PGD, and DeepFool transferred from a ResNet18 surrogate. AutoAttack and
  Carlini–Wagner attacks are not evaluated.
- **Self-referential adversarial evaluation.** BioMedAttack-LLM and SecureMed-LLM
  share the same training distribution, potentially overestimating robustness.
- **No clinical expert validation.** IDS-LLM rules were authored by the research
  team; no board-certified radiologist review of generated reports was performed.
- **Encryption scope.** Only the final validated report is encrypted. Intermediate
  embeddings and model activations are unencrypted. The inference server is assumed
  to be a trusted, isolated environment.
- **No formal regulatory compliance verification.** Components are technically
  consistent with HIPAA/GDPR PHI de-identification and encrypted transmission
  requirements but have not been verified by a compliance authority.
- **Hardware-dependent latency.** The 6.5 s inference figure was measured on a
  specific CPU configuration; results will vary with hardware, image complexity,
  and report length.

---

## 15. Module Dependency Graph

```
scripts/train.py
    ├── src/models/pipeline.py
    │       ├── src/models/biomed_clip_encoder.py
    │       └── src/models/t5_decoder.py
    ├── src/privacy/dp_training.py          (phase 1: DP-SGD)
    ├── src/adversarial/adv_training.py     (phase 2: adversarial fine-tuning)
    │       └── src/adversarial/attack_generator.py
    ├── src/anonymization/medguard_image.py
    ├── src/anonymization/medguard_text.py
    └── src/utils/data_loader.py

scripts/inference.py
    ├── src/models/pipeline.py
    ├── src/anonymization/medguard_image.py
    ├── src/anonymization/medguard_text.py
    ├── src/validation/ids_llm.py
    └── src/encryption/encrypt_report.py

scripts/evaluate.py
    ├── src/models/pipeline.py
    ├── src/anonymization/medguard_image.py
    ├── src/utils/data_loader.py
    └── src/utils/metrics.py
            └── src/anonymization/medguard_text.py  (PHI leakage via Presidio)

src/encryption/decrypt_report.py
    └── src/encryption/encrypt_report.py  (shared key utilities + constants)
```
