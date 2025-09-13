import re
import os
from enum import Enum

import nltk
import torch
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score


class NLGTaskType(Enum):
    SUMMARY = 1
    TRANSLATION = 2


def _load_nltk_data():
    # Use a cached directory
    local_data_dir = os.environ.get(
        "NLTK_DATA",
        os.path.expanduser('~/nltk_data'),
    )
    # Add the local directory to NLTK's data path
    if local_data_dir not in nltk.data.path:
        nltk.data.path.append(local_data_dir)
    # List of NLTK packages to check and download
    packages = ['punkt', 'wordnet', 'omw-1.4']
    package_paths = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    # Check for each package and download if not found
    for package in packages:
        try:
            nltk.data.find(package_paths[package])
        except LookupError:
            print(f"'{package}' not found. Downloading...")
            nltk.download(package, download_dir=local_data_dir, quiet=True)
            print(f"'{package}' downloaded successfully to '{local_data_dir}'.")


SUMMARY_PREFIX_RE = re.compile(
    r"^\s*(summary\s*:)\s*",
    flags=re.IGNORECASE
)
BERTSCORE_LANG  = "en"
BERTSCORE_MODEL = os.environ.get("BERT_SCORE_MODEL", None)


def normalize(
    s: str,
    strip_task_prefix: bool = True,
    task_type: NLGTaskType = NLGTaskType.SUMMARY
) -> str:
    if not isinstance(s, str):
        return ""
    if strip_task_prefix:
        s = s.strip()
        if task_type is NLGTaskType.SUMMARY:
            s = SUMMARY_PREFIX_RE.sub("", s)
    return s.strip()


# -------- METEOR (Explicit Tokenization) --------
_METEOR_WARNED = False
def safe_meteor(ref: str, hyp: str) -> float:
    """
    Explicit whitespace tokenization to avoid Punkt dependency;
    only warn once on exception and return 0.0.
    """
    global _METEOR_WARNED
    try:
        if not ref or not hyp:
            return 0.0
        ref_tokens = ref.lower().strip().split()
        hyp_tokens = hyp.lower().strip().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return float(meteor_score([ref_tokens], hyp_tokens))
    except Exception as e:
        if not _METEOR_WARNED:
            print(f"[WARN] METEOR calculation failed (shown only once): {e}. Continuing with a score of 0.0.")
            _METEOR_WARNED = True
        return 0.0


def compute_meteor_batch(cands, refs):
    _load_nltk_data()
    return [safe_meteor(r, c) for c, r in zip(cands, refs)]


# -------- ROUGE --------
def compute_rouge_batch(cands, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for c, r in zip(cands, refs):
        if not c or not r:
            r1.append(0.0); r2.append(0.0); rL.append(0.0)
            continue
        sc = scorer.score(r, c)  # Note the order: (ref, cand)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)
    return r1, r2, rL


def _get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- BERTScore --------
def compute_bertscore_batch(
    cands,
    refs,
    lang=BERTSCORE_LANG,
    model=BERTSCORE_MODEL,
    batch_size=64,
    device=None,
):
    device = device or _get_default_device()
    if BERTSCORE_MODEL is not None \
        and "roberta-large" in BERTSCORE_MODEL.lower():
        num_layers = 17
    else:
        num_layers = None  # Let huggingface handle it
    P, R, F1 = bert_score(
        cands,
        refs,
        lang=lang,
        model_type=model,
        batch_size=batch_size,
        verbose=False,
        device=device,
        num_layers=num_layers,
    )
    return [float(x) for x in P], [float(x) for x in R], [float(x) for x in F1]


# -------- BLEU (SacreBLEU, per-sample) --------
def compute_bleu_batch(cands, refs, tokenize="13a", smooth_method="exp"):
    r"""sentence bleu"""
    scores = []
    for hyp, ref in zip(cands, refs):
        if not hyp or not ref:
            scores.append(0.0)
            continue
        # sacrebleu.sentence_bleu accepts hypothesis(str), references(List[str])
        sb = sacrebleu.sentence_bleu(
            hypothesis=hyp,
            references=[ref],
            smooth_method=smooth_method,
            use_effective_order=True,
            lowercase=False,
            tokenize=tokenize
        )
        scores.append(float(sb.score) / 100.0)  # Convert to 0-1 range
    return scores
