import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model.eval()
if torch.cuda.is_available():
    model.to("cuda")
else:
    model.to("cpu")

# Set the pad token to the end of sentence token
model.config.pad_token_id = model.config.eos_token_id

def generate_text(prompt, max_length=50):
    """
    Generate text using the GPT-2 model.

    Args:
        prompt (str): The input text to generate from.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs,
                             max_length=max_length,
                             no_repeat_ngram_size=2,
                             temperature=0.7,
                             top_k=50,
                             top_p=0.95,
                             do_sample=True
                             )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def generate_response(prompt):
    """
    Generate a response to the given prompt.

    Args:
        prompt (str): The input text to generate a response for.

    Returns:
        str: The generated response.
    """
    response = generate_text(prompt)
    return response

if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        response = generate_response(prompt)
        print(f"DM: {response}")
        if prompt.lower() == "exit":
            print("Exiting the chatbot.")
            break