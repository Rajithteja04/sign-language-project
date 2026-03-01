"""T5-based gloss-to-English correction."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


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
        if not gloss or not gloss.strip():
            return ""

        # Expected runtime input: uppercase gloss tokens separated by spaces.
        gloss_text = " ".join(gloss.strip().split()).upper()
        self._ensure_loaded()

        if not self._ready:
            return gloss_text.capitalize() + "."

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
            text = gloss_text

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


def correct_text(raw_gloss: str) -> str:
    return gloss_to_sentence(raw_gloss)
