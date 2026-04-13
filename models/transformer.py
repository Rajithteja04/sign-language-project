"""Module-3 semantic correction: gloss tokens -> English sentence."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Optional

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForSeq2SeqLM = None
AutoTokenizer = None

MSASL_VOCAB = {"COUSIN", "EAT", "FINISH", "NICE", "TEACHER"}

TOKEN_SANITIZE_RE = re.compile(r"[^A-Z0-9]+")


def _canonicalize_subset_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for tok in tokens:
        norm = TOKEN_SANITIZE_RE.sub("_", str(tok).upper()).strip("_")
        if norm:
            cleaned.append(norm)
    return tuple(sorted(cleaned))


def _build_subset_sentence_map() -> dict[tuple[int, tuple[str, ...]], str]:
    # Deterministic templates for LSA64 subset combinations
    rows = [
        (("HELP", "BUY"), "Can you help me buy this?"),
        (("GIVE", "NAME"), "Please give me your name."),
        (("GIVE", "RICE"), "Please give me some rice."),
        (("THANKS", "HELP"), "Thanks for all your help."),
        (("BUY", "MILK"), "I need to buy some milk."),
        (("WHERE", "FOOD"), "Where is the food kept?"),
        (("NAME", "WHERE"), "Where is [Name] located?"),
        (("WHERE", "WATER"), "Where can I find some water?"),
        (("WHERE", "BUY", "RICE"), "Where can I buy some rice?"),
        (("GIVE", "WATER", "THANKS"), "Thanks for giving me the water."),
        (("HELP", "WHERE", "FOOD"), "Can you help me find where the food is?"),
        (("BUY", "MILK", "RICE"), "I want to buy milk and rice."),
        (("HELP", "NAME", "THANKS"), "Thanks for helping me, [Name]."),
        (("GIVE", "FOOD", "WATER"), "Please give them food and water."),
        (("NAME", "WHERE", "GIVE"), "Where should I give this to [Name]?"),
        (("GIVE", "HELP", "MILK"), "Can you help me give the baby milk?"),
    ]

    mapping: dict[tuple[int, tuple[str, ...]], str] = {}
    for tokens, sentence in rows:
        key_tokens = _canonicalize_subset_tokens(tokens)
        if not key_tokens:
            continue
        mapping[(len(key_tokens), key_tokens)] = sentence.strip()
    return mapping


LSA64_SUBSET_SENTENCES = _build_subset_sentence_map()


def _load_lsa64_metadata() -> dict[str, dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "data" / "lsa64_labels.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for key, meta in raw.items():
        norm = re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")
        if not norm:
            continue
        if isinstance(meta, dict):
            phrase = meta.get("phrase") or key.replace("_", " ").lower()
            role = meta.get("role", "object")
            progressive = meta.get("progressive")
        else:
            phrase = str(meta)
            role = "object"
            progressive = None
        entry: dict[str, str] = {
            "phrase": phrase,
            "role": role,
        }
        if progressive:
            entry["progressive"] = progressive
        normalized[norm] = entry
    return normalized


LSA64_METADATA = _load_lsa64_metadata()
LSA64_VOCAB = set(LSA64_METADATA.keys())


class TransformerCorrector:
    def __init__(self, model_name: str = "t5-small") -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._ready = False
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._ready or self._load_error:
            return
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            self._load_error = "transformers or torch is not installed"
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            # Keep inference CPU-friendly by default.
            self.model.to("cpu")
            self.model.eval()
            self._ready = True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)

    def correct(self, gloss: str) -> str:
        tokens = normalize_gloss_tokens(gloss)
        return self.correct_tokens(tokens)

    def correct_tokens(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        token_set = set(tokens)

        normalized_tokens = [_sanitize_token(tok) for tok in tokens]
        key = (len(normalized_tokens), tuple(sorted(normalized_tokens)))
        subset_sentence = LSA64_SUBSET_SENTENCES.get(key)
        if subset_sentence:
            if not subset_sentence.endswith((".", "?", "!")):
                subset_sentence += "."
            return subset_sentence

        if token_set.issubset(MSASL_VOCAB):
            return _msasl_fallback(tokens)

        if LSA64_VOCAB and token_set.issubset(LSA64_VOCAB):
            return _lsa64_fallback(tokens)

        gloss_text = " ".join(tokens)
        self._ensure_loaded()

        if not self._ready:
            return _generic_fallback(tokens)

        prompt = f"translate gloss to english: {gloss_text}"

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not text:
            return _generic_fallback(tokens)

        # If t5-small output is noisy, prompt-leaked, or gloss-echoed, use fallback.
        if text.upper().strip(".!?") == gloss_text or _looks_invalid_output(text):
            if LSA64_VOCAB and token_set.issubset(LSA64_VOCAB):
                return _lsa64_fallback(tokens)
            return _generic_fallback(tokens)

        text = text[0].upper() + text[1:] if text else ""
        if text and text[-1] not in ".!?":
            text += "."

        return text


_singletons: dict[str, TransformerCorrector] = {}


def _get_singleton(model_name: str = "t5-small") -> TransformerCorrector:
    corrector = _singletons.get(model_name)
    if corrector is None:
        corrector = TransformerCorrector(model_name=model_name)
        _singletons[model_name] = corrector
    return corrector


def gloss_to_sentence(gloss: str) -> str:
    """
    Convert uppercase gloss tokens into a corrected English sentence.
    """
    return _get_singleton("t5-small").correct(gloss)


def words_to_sentence(words: list[str]) -> str:
    """
    Module-3 entrypoint for runtime:
    takes committed word list and returns one corrected sentence.
    """
    if not words:
        return ""
    tokens: list[str] = []
    for word in words:
        token = _sanitize_token(word)
        if token:
            tokens.append(token)
    return _get_singleton("t5-small").correct_tokens(tokens)


def correct_text(raw_gloss: str) -> str:
    return gloss_to_sentence(raw_gloss)


def normalize_gloss_tokens(gloss: str) -> list[str]:
    """
    Normalize gloss to stable uppercase tokens and remove immediate repeats.
    """
    if not gloss:
        return []
    gloss = gloss.replace("-", "_")
    raw = re.findall(r"[A-Za-z_']+", gloss.upper())
    if not raw:
        return []
    cleaned: list[str] = []
    for tok in raw:
        tok = tok.strip("_")
        if not tok:
            continue
        if cleaned and cleaned[-1] == tok:
            continue
        cleaned.append(tok)
    return cleaned


def _sanitize_token(word: str) -> str:
    return TOKEN_SANITIZE_RE.sub("_", word.upper()).strip("_")


def _msasl_fallback(tokens: list[str]) -> str:
    """
    Lightweight deterministic fallback for demo stability when t5-small output is weak.
    """
    token_set = set(tokens)

    has_cousin = "COUSIN" in token_set
    has_teacher = "TEACHER" in token_set
    has_eat = "EAT" in token_set
    has_finish = "FINISH" in token_set
    has_nice = "NICE" in token_set

    # Pick subject first so every combination can preserve key words.
    if has_cousin and has_teacher:
        subject = "My cousin and the teacher"
    elif has_cousin:
        subject = "My cousin"
    elif has_teacher:
        subject = "The teacher"
    else:
        subject = "I"

    if has_eat and has_finish:
        text = f"{subject} finished eating"
        if has_nice:
            text += ", and it was nice"
        return text + "."

    if has_eat:
        if subject == "I":
            text = "I want to eat"
        else:
            text = f"{subject} is eating"
        if has_nice:
            text += ", and it is nice"
        return text + "."

    if has_finish:
        text = "I finished" if subject == "I" else f"{subject} finished"
        if has_nice:
            text += ", and it was nice"
        return text + "."

    if has_nice:
        text = "It is nice" if subject == "I" else f"{subject} is nice"
        return text + "."

    # Noun-only combos (no verb cues) -> use a natural copula sentence.
    if has_cousin and has_teacher:
        return "My cousin is a teacher."
    if has_teacher:
        return "The teacher is here."
    if has_cousin:
        return "My cousin is here."

    if len(tokens) == 1:
        single = tokens[0]
        if single == "COUSIN":
            return "My cousin."
        if single == "TEACHER":
            return "The teacher."
        return single.capitalize() + "."

    text = " ".join(t.lower() for t in tokens).capitalize()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _generic_fallback(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tok.capitalize() for tok in tokens if tok)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _lsa64_fallback(tokens: list[str]) -> str:
    if not tokens:
        return ""

    seen: set[tuple[str, str]] = set()
    buckets: dict[str, list[dict[str, str]]] = {}

    for tok in tokens:
        key = _sanitize_token(tok)
        meta = LSA64_METADATA.get(key)
        if not meta:
            continue
        phrase = meta.get("phrase")
        role = meta.get("role", "object")
        if not phrase:
            continue
        sig = (role, phrase.lower())
        if sig in seen:
            continue
        seen.add(sig)
        buckets.setdefault(role, []).append(meta)

    if not buckets:
        return _generic_fallback(tokens)

    objects = [m["phrase"] for m in buckets.get("object", [])]
    subjects = [m["phrase"] for m in buckets.get("subject", [])]
    adjectives = [m["phrase"] for m in buckets.get("adjective", [])]
    verbs = buckets.get("verb", [])
    questions = buckets.get("question", [])

    subject_phrase = _join_phrases(subjects)
    borrowed_subject = False
    if not subject_phrase and not verbs and objects:
        subject_phrase = objects.pop(0)
        borrowed_subject = True
    object_phrase = _join_phrases(objects)

    if questions:
        return _lsa64_question_sentence(questions[0], subject_phrase, object_phrase)

    sentences: list[str] = []

    if verbs:
        sentences.append(
            _lsa64_verb_sentence(
                subject_phrase or "I",
                object_phrase,
                verbs[0],
            )
        )

    if adjectives:
        target = object_phrase or subject_phrase or "it"
        sentences.append(_lsa64_adjective_sentence(target, adjectives))

    if not sentences:
        noun_phrase = subject_phrase or object_phrase
        if noun_phrase:
            copula = "are" if _is_plural_phrase(noun_phrase) else "is"
            sentences.append(f"{_capitalize_phrase(noun_phrase)} {copula} detected.")
        else:
            fallback = ", ".join(tok.capitalize() for tok in tokens if tok)
            if not fallback:
                fallback = "Detected gesture"
            sentences.append(f"{fallback}.")

    return " ".join(sentences)


def _lsa64_question_sentence(
    question_meta: dict[str, str],
    subject_phrase: Optional[str],
    object_phrase: Optional[str],
) -> str:
    prompt = (question_meta.get("phrase") or "where").strip()
    target = object_phrase or subject_phrase or "it"
    target = target.strip()
    if not target:
        target = "it"
    if prompt.lower() == "where":
        copula = "are" if _is_plural_phrase(target) else "is"
        sentence = f"Where {copula} {target}?".replace("  ", " ").strip()
    else:
        sentence = f"{_capitalize_phrase(prompt)} {target}?".replace("  ", " ").strip()
    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def _lsa64_verb_sentence(
    subject_phrase: str,
    object_phrase: Optional[str],
    verb_meta: dict[str, str],
) -> str:
    subject = subject_phrase or "I"
    copula = _copula_for_subject(subject)
    progressive = _verb_progressive(verb_meta)
    base_verb = verb_meta.get("phrase", "act")

    if progressive:
        clause = f"{_capitalize_phrase(subject)} {copula} {progressive}"
    else:
        clause = f"{_capitalize_phrase(subject)} {base_verb}"

    if object_phrase:
        clause = f"{clause} {object_phrase}"

    clause = clause.strip()
    if clause and clause[-1] not in ".!?":
        clause += "."
    return clause


def _lsa64_adjective_sentence(target_phrase: str, adjectives: list[str]) -> str:
    target = target_phrase or "it"
    adj_text = _join_phrases(adjectives)
    copula = _copula_for_subject(target)
    sentence = f"{_capitalize_phrase(target)} {copula} {adj_text}."
    return sentence


def _join_phrases(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _capitalize_phrase(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _copula_for_subject(text: str) -> str:
    lowered = (text or "").strip().lower()
    if lowered == "i":
        return "am"
    if lowered in {"you", "we", "they"}:
        return "are"
    if _is_plural_phrase(lowered):
        return "are"
    return "is"


def _is_plural_phrase(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if " and " in lowered:
        return True
    plural_keywords = {
        "colors",
        "women",
        "men",
        "people",
        "countries",
        "sons",
        "daughters",
    }
    last_word = lowered.split()[-1]
    if last_word in plural_keywords:
        return True
    if last_word.endswith("s") and not last_word.endswith("ss"):
        return True
    return False


def _verb_progressive(meta: dict[str, str]) -> str:
    if not meta:
        return ""
    if "progressive" in meta and meta["progressive"]:
        return meta["progressive"]
    phrase = meta.get("phrase", "")
    if not phrase:
        return ""
    words = phrase.split()
    words[-1] = _ing_form(words[-1])
    return " ".join(words)


def _ing_form(word: str) -> str:
    if not word:
        return ""
    base = word.lower()
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("ee"):
        return base + "ing"
    if len(base) > 2 and _is_cvc(base):
        return base + base[-1] + "ing"
    if base.endswith("e"):
        return base[:-1] + "ing"
    if base.endswith("ing"):
        return base
    return base + "ing"


def _is_cvc(word: str) -> bool:
    vowels = "aeiou"
    if len(word) < 3:
        return False
    a, b, c = word[-3], word[-2], word[-1]
    return (a not in vowels) and (b in vowels) and (c not in vowels and c not in "wy")


def _looks_invalid_output(text: str) -> bool:
    low = text.lower()
    bad_markers = (
        "glossary",
        "return one short",
        "translate gloss",
        "english sentence only",
    )
    if any(marker in low for marker in bad_markers):
        return True
    if len(low.split()) > 30:
        return True
    return False
