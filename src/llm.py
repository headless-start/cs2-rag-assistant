"""Generator abstraction.

One class, two providers, chosen by ``LLM_PROVIDER``:

* ``hf``     — a local ``transformers`` instruct model (default), so the project
               runs with no paid API.
* ``openai`` — any OpenAI-compatible ``/chat/completions`` endpoint, for swapping
               in a bigger hosted model with one env change.

The model loads lazily on first ``generate``.
"""
import os

from .config import settings, resolve_device


class Generator:
    def __init__(self, provider=None, model=None):
        self.provider = provider or settings.llm_provider
        self.model_name = model or settings.gen_model
        self._model = None
        self._tokenizer = None

    # -- local transformers ---------------------------------------------------
    def _load_hf(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = resolve_device(settings.gen_device)
        kwargs = {"torch_dtype": "auto"}
        if os.environ.get("LLM_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
            kwargs["device_map"] = "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            self._model.to(device)
        self._device = device

    def _generate_hf(self, system, user, max_new_tokens, temperature):
        self._load_hf()
        import torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    # -- openai-compatible endpoint ------------------------------------------
    def _generate_openai(self, system, user, max_new_tokens, temperature):
        import requests
        url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def generate(self, system, user, max_new_tokens=None, temperature=0.0):
        max_new_tokens = max_new_tokens or settings.max_new_tokens
        if self.provider == "openai":
            return self._generate_openai(system, user, max_new_tokens, temperature)
        return self._generate_hf(system, user, max_new_tokens, temperature)
