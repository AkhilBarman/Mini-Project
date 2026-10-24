"""
LLM Sampling Module for Approach B (Self-Consistency / SelfCheckGPT).

Provides utilities to generate or load multiple stochastic summary samples
for court judgments using local Ollama, Hugging Face transformers, or cached files.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error


def compute_prompt_hash(text: str) -> str:
    """Return MD5 hex digest of normalised prompt text."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


class LLMSampler:
    """
    Samples multiple stochastic summaries for a source judgment.
    
    Supported backends:
    - 'ollama': Queries a local Ollama instance (e.g. llama3, llama3.2).
    - 'transformers': Generates samples via Hugging Face pipeline on GPU/CPU.
    - 'mock': Deterministic generator for unit tests and offline testing.
    """

    def __init__(
        self,
        backend: str = "mock",
        model_name: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        pipeline_instance: Optional[Any] = None
    ) -> None:
        """
        Initialize LLMSampler.

        Args:
            backend: 'ollama', 'transformers', or 'mock'.
            model_name: Model identifier (e.g., 'llama3', 'meta-llama/Llama-3.2-1B-Instruct').
            ollama_url: Base URL for Ollama REST API.
            temperature: Sampling temperature (higher = more stochastic diversity).
            max_tokens: Max new tokens to generate.
            pipeline_instance: Optional pre-initialized transformers pipeline.
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pipeline = pipeline_instance

        if self.backend == "transformers" and self.pipeline is None:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )
            except Exception as e:
                print(f"Warning: Could not initialize transformers pipeline: {e}")
                self.pipeline = None

    def build_prompt(self, judgment_text: str) -> str:
        """Build standard summarisation prompt for court judgments."""
        # Truncate judgment text to keep prompt within standard context window
        truncated_judgment = judgment_text[:4000].strip()
        prompt = (
            "You are a legal summarisation assistant. Read the following Indian court judgment "
            "and write an objective, factual summary of the holding, key facts, and legal reasoning.\n\n"
            f"JUDGMENT:\n{truncated_judgment}\n\n"
            "SUMMARY:"
        )
        return prompt

    def sample_summaries(
        self,
        judgment_text: str,
        n_samples: int = 3
    ) -> List[str]:
        """
        Generate n_samples stochastic summaries for the judgment text.

        Returns:
            List[str]: list of n_samples summary texts.
        """
        if self.backend == "mock":
            return self._sample_mock(judgment_text, n_samples)
        elif self.backend == "ollama":
            return self._sample_ollama(judgment_text, n_samples)
        elif self.backend == "transformers":
            return self._sample_transformers(judgment_text, n_samples)
        else:
            raise ValueError(f"Unknown sampling backend: {self.backend}")

    def _sample_mock(self, judgment_text: str, n_samples: int) -> List[str]:
        """Generate deterministic mock samples derived from judgment text for testing."""
        first_chunk = judgment_text[:300].strip()
        samples = []
        for i in range(n_samples):
            samples.append(
                f"[Sample {i+1}] In this case, the High Court considered the submissions of the parties. "
                f"Holding context: {first_chunk}. Relief was granted according to law."
            )
        return samples

    def _sample_ollama(self, judgment_text: str, n_samples: int) -> List[str]:
        """Query local Ollama server to generate stochastic samples."""
        prompt = self.build_prompt(judgment_text)
        url = f"{self.ollama_url}/api/generate"
        samples = []

        for _ in range(n_samples):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    samples.append(res_body.get("response", "").strip())
            except Exception as e:
                print(f"Error querying Ollama API at {url}: {e}")
                # Fallback to empty string on network failure
                samples.append("")

        return samples

    def _sample_transformers(self, judgment_text: str, n_samples: int) -> List[str]:
        """Generate samples using local/Colab Hugging Face text-generation pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Transformers text-generation pipeline is not initialized.")

        prompt = self.build_prompt(judgment_text)
        outputs = self.pipeline(
            prompt,
            num_return_sequences=n_samples,
            do_sample=True,
            temperature=self.temperature
        )
        samples = []
        for out in outputs:
            gen_text = out["generated_text"]
            # Strip prompt from generated response
            if gen_text.startswith(prompt):
                gen_text = gen_text[len(prompt):].strip()
            samples.append(gen_text)
        return samples


def save_sampled_summaries(
    pair_id: str,
    samples: List[str],
    output_dir: str = "data/sampled_summaries"
) -> str:
    """Save generated samples for a given pair_id to disk as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{pair_id}_samples.json")
    payload = {
        "pair_id": pair_id,
        "n_samples": len(samples),
        "samples": samples
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath


def load_sampled_summaries(
    pair_id: str,
    input_dir: str = "data/sampled_summaries"
) -> List[str]:
    """Load previously saved summary samples for a pair_id from disk."""
    filepath = os.path.join(input_dir, f"{pair_id}_samples.json")
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception:
        return []
