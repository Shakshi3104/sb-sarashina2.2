import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import Literal

from timer import stop_watch


@stop_watch
def generate_text(input_text: str, weights_version: Literal["0.5b", "1b", "3b"] = "3b") -> str:
    model_name = f"sbintuitions/sarashina2.2-{weights_version}-instruct-v0.1"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        model.to(device)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                         device="mps" if torch.backends.mps.is_available() else "cpu")

    messages = [
        {"role": "user", "content": f"{input_text}"},
    ]

    outputs = generator(
        messages,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

    return outputs[-1]["generated_text"][-1]["content"]


if __name__ == "__main__":

    input_text = "なにわ男子について教えてください"

    output = generate_text(input_text, weights_version="3b")
    print(output)
