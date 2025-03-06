import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from timer import stop_watch


@stop_watch
def generate_text(input_text: str) -> str:
    # TODO: Replace model_name
    model_name = "cyberagent/open-calm-small"

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

    input_text = "たまごっちとは何ですか？"

    output = generate_text(input_text)
    print(output)
