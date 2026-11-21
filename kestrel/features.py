"""Step 3 feature extraction from the base model.

The Qwen backend imports torch/transformers lazily, so this module and its
tests run in CI without model dependencies. Real extraction runs on the RTX
3050 machine where the 4-bit model is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kestrel.verifier import VerifierFeatures

__all__ = [
    "DEFAULT_MODEL",
    "GeneratedAnswer",
    "ModelBackend",
    "QwenBackend",
    "to_features",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class GeneratedAnswer:
    """One answer plus the signals needed by the verifier."""

    text: str
    hidden: tuple[float, ...]
    token_logprobs: tuple[float, ...]
    sampled_answers: tuple[str, ...] = ()


def to_features(answer: GeneratedAnswer) -> VerifierFeatures:
    """Map a generated answer to the verifier's feature contract.

    Stated confidence is unavailable for this base model, so it stays None and
    the baseline returns 0.5.
    """
    return VerifierFeatures(
        last_token_hidden=answer.hidden,
        token_logprobs=answer.token_logprobs,
        sampled_answers=answer.sampled_answers,
    )


class ModelBackend(Protocol):
    """Seam for answer generation; fake backends make training testable."""

    def generate(
        self, prompt: str, n_samples: int = 1, max_new_tokens: int = 256
    ) -> GeneratedAnswer:
        """Generate an answer and its verifier signals."""
        ...


class QwenBackend:
    """Qwen2.5-1.5B-Instruct in 4-bit NF4, per the Step 2 handoff.

    Hidden features come from a fresh forward pass over the generated sequence:
    last layer, last token, dimension 1536. Token log-probabilities are gathered
    from the generation logits. Sampled answers provide self-consistency.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        n_samples: int = 3,
        max_new_tokens: int = 256,
        seed: int = 20260103,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.manual_seed(self.seed)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=config,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._torch = torch

    def generate(
        self,
        prompt: str,
        n_samples: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GeneratedAnswer:
        """Generate one answer and extract its verifier signals."""
        self._load()
        torch = self._torch
        samples = self.n_samples if n_samples is None else n_samples
        new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if samples < 1 or new_tokens < 1:
            raise ValueError("n_samples and max_new_tokens must be positive")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                do_sample=samples > 1,
                num_return_sequences=samples,
                max_new_tokens=new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Last-layer, last-token hidden state from a clean pass over the output.
        with torch.no_grad():
            full = self._model(
                input_ids=generated.sequences[:1], output_hidden_states=True
            )
        hidden = full.hidden_states[-1][0, -1, :].float().cpu().tolist()

        # Log-probability of each generated token in the primary sample.
        logprobs = []
        for step, logits in enumerate(generated.scores):
            token_id = generated.sequences[0, prompt_len + step]
            log_p = torch.log_softmax(logits.float(), dim=-1).gather(0, token_id)
            logprobs.append(float(log_p.cpu()))

        texts = tuple(
            self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in generated.sequences
        )
        return GeneratedAnswer(
            text=texts[0],
            hidden=tuple(hidden),
            token_logprobs=tuple(logprobs),
            sampled_answers=texts,
        )
