import re


class TransformerCorrector:
    """
    Wrapper around a HuggingFace text2text Transformer model.
    Falls back to lightweight cleanup if transformers is unavailable.
    """

    def __init__(self, model_name: str | None = None, device: int = -1):
        self.model_name = model_name
        self._pipeline = None
        if model_name:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    task="text2text-generation",
                    model=model_name,
                    device=device,
                )
            except Exception:
                self._pipeline = None

    @staticmethod
    def _fallback_cleanup(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        return text

    def correct(self, raw_gloss: str) -> str:
        if self._pipeline is None:
            return self._fallback_cleanup(raw_gloss)
        try:
            out = self._pipeline(
                f"Fix grammar and rewrite naturally: {raw_gloss}",
                max_length=96,
                num_beams=4,
            )
            if out and isinstance(out, list) and "generated_text" in out[0]:
                return self._fallback_cleanup(out[0]["generated_text"])
        except Exception:
            pass
        return self._fallback_cleanup(raw_gloss)


def correct_text(raw_gloss: str, model_name: str | None = None) -> str:
    return TransformerCorrector(model_name=model_name).correct(raw_gloss)
