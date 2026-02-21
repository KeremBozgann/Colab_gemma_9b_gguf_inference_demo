from llama_cpp import Llama

# Define the inference parameters
inference_params = {
    "n_threads": 4,
    "n_predict": -1,
    "top_k": 20,
    "min_p": 0.0,
    "top_p": 0.95,
    "temp": 0.01,
    "repeat_penalty": 1.05,
    "input_prefix": "<start_of_turn>user\\n",
    "input_suffix": "<end_of_turn>\\n<start_of_turn>model\\n",
    "antiprompt": [],
    "pre_prompt": "",
    "pre_prompt_suffix": "",
    "pre_prompt_prefix": "<bos>",
    "seed": -1,
    "tfs_z": 1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n_keep": 0,
    "logit_bias": {},
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "memory_f16": True,
    "multiline_input": False,
    "penalize_nl": True
}

# Initialize the Gemma model with the specified inference parameters
gemma = Llama.from_pretrained(
    repo_id="ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=False,
    n_ctx=4096
)
# Example input
user_input = "Türkiyenin başkenti neresidir?"

# Construct the prompt
prompt = f"{inference_params['pre_prompt_prefix']}{inference_params['pre_prompt']}{inference_params['pre_prompt_suffix']}{inference_params['input_prefix']}{user_input}{inference_params['input_suffix']}"

# Generate the response
response = gemma(prompt, max_tokens=2048, temperature=0.1, top_p=0.9,)

# Output the response
print(response['choices'][0]['text'])
