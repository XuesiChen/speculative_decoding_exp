import time
import pandas as pd
from vllm import LLM, SamplingParams, CompletionOutput

# 10 long prompts
prompts = [
    "Explain the history and future of artificial general intelligence in detail.",
    "Describe the evolution of deep learning models from 2010 to today, including major milestones.",
    "Write a detailed summary of how neural networks are trained, including optimizers and loss functions.",
    "Explain the full pipeline of deploying a large language model in production, from training to inference.",
    "Give a comprehensive overview of transformers and how they revolutionized natural language processing.",
    "Discuss the ethical implications of using AI in healthcare with examples and mitigation strategies.",
    "Describe how reinforcement learning works, including examples and mathematical formulations.",
    "Write a long essay on the societal impacts of large-scale AI deployment, both positive and negative.",
    "Explain how multimodal models work and what challenges they solve, with technical depth.",
    "Provide an end-to-end explanation of how search engines rank web pages using machine learning."
]

# Sweep speculative lengths
speculative_token_lengths = [2, 4, 8, 16, 32, 64, 128]
draft_model = "facebook/opt-125m"
target_model = "facebook/opt-6.7b"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

summary_rows = []
detailed_rows = []

for num_tokens in speculative_token_lengths:
    print(f"Running with num_speculative_tokens = {num_tokens}")

    llm = LLM(
        model=target_model,
        tensor_parallel_size=1,
        speculative_config={
            "model": draft_model,
            "num_speculative_tokens": num_tokens,
        },
    )

    ttft_list = []
    tbt_list = []
    total_list = []
    input_lens = []
    output_lens = []

    for prompt in prompts:
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        first_token_time = time.time()
        output: CompletionOutput = outputs[0].outputs[0]
        end = time.time()

        num_input_tokens = len(outputs[0].prompt_token_ids)
        num_output_tokens = len(output.token_ids)

        if num_output_tokens == 0:
            continue

        ttft = first_token_time - start
        tbt = (end - first_token_time) / num_output_tokens
        total = end - start

        # Save detailed data
        detailed_rows.append({
            "speculative_tokens": num_tokens,
            "draft_model": draft_model,
            "target_model": target_model,
            "input_tokens": num_input_tokens,
            "output_tokens": num_output_tokens,
            "ttft_sec(s)": round(ttft, 10),
            "tbt_sec_per_token(s)": round(tbt, 10),
            "total_latency_sec(s)": round(total, 10),
            "prompt": prompt
        })

        # For summary
        ttft_list.append(ttft)
        tbt_list.append(tbt)
        total_list.append(total)
        input_lens.append(num_input_tokens)
        output_lens.append(num_output_tokens)

    summary_rows.append({
        "speculative_tokens": num_tokens,
        "draft_model": draft_model,
        "target_model": target_model,
        "avg_input_tokens": round(sum(input_lens) / len(input_lens), 10),
        "avg_output_tokens": round(sum(output_lens) / len(output_lens), 10),
        "avg_ttft_sec(s)": round(sum(ttft_list) / len(ttft_list), 10),
        "avg_tbt_sec_per_token(s)": round(sum(tbt_list) / len(tbt_list), 10),
        "avg_total_latency_sec(s)": round(sum(total_list) / len(total_list), 10)
    })

    del llm
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print("finished processing experiments for speculative token length of ", num_tokens)

print("Done running experiments")

# Write CSVs
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("specdecode_summary.csv", index=False)

detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv("specdecode_detailed.csv", index=False)

print("✅ Done. Saved summary to 'specdecode_summary.csv' and details to 'specdecode_detailed.csv'")
