import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

# ---------------------------------------------------------------------------
# Device selection — prefer Apple Silicon MPS, fall back to CUDA then CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_model(model_id: str = MODEL_ID):
    print(f"Loading model '{model_id}' on device '{DEVICE}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    )
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def generate_reply(
    tokenizer,
    model,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Run one forward pass and return the assistant's reply text.

    `conversation` is a list of {"role": "user"|"assistant"|"system", "content": "…"} dicts.
    """
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return reply.strip()


def chat_loop(tokenizer, model):
    print("=" * 60)
    print("  SmolLM3-3B  —  CLI Chat  (type 'exit' or 'quit' to stop)")
    print("=" * 60)

    system_prompt = (
        "You are a helpful, concise, and honest AI assistant."
    )
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        reply = generate_reply(tokenizer, model, conversation)
        print(reply)

        conversation.append({"role": "assistant", "content": reply})


def main():
    tokenizer, model = load_model()
    chat_loop(tokenizer, model)


if __name__ == "__main__":
    main()
