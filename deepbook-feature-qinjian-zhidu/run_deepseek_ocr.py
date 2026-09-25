from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "deepseek-ai/DeepSeek-OCR-2"
image_file = r"C:\Users\Lenovo\Desktop\毕设开发\屏幕截图 2025-11-14 164431.png"
output_path = r"C:\Users\Lenovo\Desktop\毕设开发\ocr_output"

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available. Check driver and torch install.")

attn_impl = "flash_attention_2"
try:
    import flash_attn  # noqa: F401
    print("flash-attn available -> using flash_attention_2")
except Exception as e:
    attn_impl = "eager"
    print(f"flash-attn not available ({e}); falling back to eager")

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("using attn_impl:", attn_impl)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "

Path(output_path).mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation=attn_impl,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True,
)
print("Done. Output saved to:", output_path)
print("Result:", res)
