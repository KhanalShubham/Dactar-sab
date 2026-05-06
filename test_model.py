from huggingface_hub import InferenceClient

import os
client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HF_TOKEN", "")
)

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": """You are a radiologist. Generate a lumbar spine MRI report.

Findings:
L1-L2: Normal
L2-L3: Mild disc degeneration
L3-L4: Posterior disc bulge
L4-L5: Moderate canal stenosis
L5-S1: Normal

Generate:
- Findings section
- Impression section
"""
        }
    ],
    max_tokens=600
)

print(response.choices[0].message.content)