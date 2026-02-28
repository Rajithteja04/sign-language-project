from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from cv2 import data
from sympy import content


# ---------------------------------------------------
# Utility Cleanup
# ---------------------------------------------------

def _fallback_cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _validate_preserved_words(raw_gloss: str, output: str) -> bool:
    """
    Ensures important words from gloss are preserved in output.
    Simple but effective validation.
    """
    gloss_words = set(raw_gloss.lower().split())
    output_words = set(re.sub(r"[^\w\s]", "", output.lower()).split())

    # At least one content word must match
    return len(gloss_words.intersection(output_words)) > 0


# ---------------------------------------------------
# Local Transformer Backend (HuggingFace)
# ---------------------------------------------------

class TransformerCorrector:
    """Local lightweight NLP corrector (HuggingFace)."""

    def __init__(self, model_name: str | None = None, device: int = -1):
        self.model_name = model_name or "google/flan-t5-base"
        self._pipeline = None

        try:
            from transformers import pipeline # type: ignore

            self._pipeline = pipeline(
                task="text2text-generation",
                model=self.model_name,
                device=device,
            )
        except Exception:
            self._pipeline = None

    def correct(self, raw_gloss: str) -> str:
        print("PIPELINE STATUS:", self._pipeline is not None)

        if self._pipeline is None:
            print("Pipeline is None")
            return _fallback_cleanup(raw_gloss)

        gloss = str(raw_gloss).strip().lower()

        prompt = (
    "You are converting American Sign Language (ASL) gloss into natural English.\n"
    "Rewrite the gloss as one complete grammatically correct English sentence.\n"
    "Add missing auxiliary verbs, articles, and prepositions if needed.\n"
    "Do not remove important words.\n\n"
    f"ASL Gloss: {gloss}\n"
    "Correct English sentence:"
)

        print("PROMPT:", prompt)

        out = self._pipeline(
            prompt,
            max_length=96,
            num_beams=6,
            do_sample=False,
        )

        print("RAW MODEL OUTPUT:", out)

        text = str(out[0]["generated_text"]).strip()
        print("CLEAN TEXT:", text)

        return _fallback_cleanup(text)


# ---------------------------------------------------
# Gemini API Backend
# ---------------------------------------------------

class GeminiAPICorrector:
    """Gemini API-based NLP corrector with local fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        timeout_s: int = 60,
    ) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = model_name
        self.timeout_s = timeout_s

    def _call_gemini(self, text: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self.model_name}:generateContent?key={urllib.parse.quote(self.api_key)}"
        )

        prompt = (
            "Convert the following ASL gloss into a complete, grammatically correct English sentence. "
            "You must include all important words from the gloss. "
            "Do not shorten the meaning. "
            "The sentence must be fully complete.\n\n"
            f"ASL gloss: {text.strip().upper()}\n\n"
            "English sentence:"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.95,
                "maxOutputTokens": 80,
            },
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)

        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError(f"No candidates in response: {data}")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "text" in part:
                return part["text"].strip()

        raise RuntimeError(f"No text found in response: {data}")

    def correct(self, raw_gloss: str) -> str:
        try:
            out = self._call_gemini(str(raw_gloss))
            print("RAW GEMINI OUTPUT:", out)

            for marker in ("Sentence:", "Output:", "Answer:", "Gloss:"):
                if marker in out:
                    out = out.split(marker, 1)[-1].strip()

            if "\n" in out:
                out = out.splitlines()[0].strip()

            out = _fallback_cleanup(out)

            if _validate_preserved_words(raw_gloss, out):
                return out
            else:
                print("Validation failed. Falling back.")

        except Exception as e:
            print("GEMINI ERROR:", e)
            raise  # <-- THIS IS IMPORTANT

        return _fallback_cleanup(raw_gloss)
# ---------------------------------------------------
# Factory Function
# ---------------------------------------------------

def create_text_corrector(
    backend: str = "transformer",
    transformer_model: str | None = None,
    gemini_model: str = "gemini-2.5-flash",
    gemini_api_key: str | None = None,
):
    backend = (backend or "transformer").strip().lower()

    if backend in {"gemini", "api"}:
        return GeminiAPICorrector(
            api_key=gemini_api_key,
            model_name=gemini_model,
        )

    return TransformerCorrector(model_name=transformer_model)


def correct_text(raw_gloss: str, model_name: str | None = None) -> str:
    return TransformerCorrector(model_name=model_name).correct(raw_gloss)