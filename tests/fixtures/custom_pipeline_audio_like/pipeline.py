"""Small authored fixture for static analysis; never imported on the host."""
from transformers import AutoModel, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY


class AudioPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]
        generation_kwargs = {k: kwargs[k] for k in kwargs if k in generation_keys}
        return {}, generation_kwargs, {}

    def preprocess(self, inputs):
        turns = inputs.get("turns", [])
        audio = inputs.get("audio", None)
        prompt = inputs.get("prompt", "Listen.")
        sampling_rate = inputs.get("sampling_rate", 16000)
        if not turns:
            turns.append({"role": "user", "content": prompt})
        return self.processor(turns=turns, audio=audio, sampling_rate=sampling_rate)

    def _forward(self, model_inputs, temperature=None, max_new_tokens=None, repetition_penalty=1.1):
        temperature = temperature or None
        do_sample = temperature is not None
        return self.model.generate(
            **model_inputs, do_sample=do_sample, temperature=temperature,
            max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
        )

    def postprocess(self, outputs):
        return self.tokenizer.decode(outputs, skip_special_tokens=True)


PIPELINE_REGISTRY.register_pipeline("listen-and-answer", pipeline_class=AudioPipeline,
                                    pt_model=AutoModel, type="multimodal")
