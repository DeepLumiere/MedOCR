```markdown
# 🩺✨ MedOCR: Mobile Lab Report Analysis with Gemma-3N & Unsloth FastModels 🚀📲

=======
![MedOCR Test Image](https://raw.githubusercontent.com/DeepLumiere/MedOCR/blob/main/labimage.png)
>>>>>>> main
---

## 🌟 Overview

**MedOCR** empowers instant, AI-driven lab report analysis by leveraging the multi-modal Gemma-3N model combined with Unsloth FastModels for efficient, fast, and accurate extraction of medical test names from lab report images. Designed for mobile and resource-limited environments, MedOCR brings timely health insights directly to users’ fingertips.

---

## 🎯 Features

- 🖼️ Multi-modal AI OCR extracts distinct medical test names from lab report images with contextual understanding.
- ⚡ 4-bit quantization and Unsloth FastModels ensure fast, memory-efficient inference.
- 📱 Optimized for mobile devices and Kaggle-style low-GPU environments.
- 🔍 Interactive test insights: values, normal ranges, and possible abnormalities.
- 💾 Support for LoRA fine-tuning and saving models and tokenizers.

---

## 🚀 Quickstart

### 1️⃣ Installation

```
pip install unsloth transformers timm
```

### 2️⃣ Load Model and Tokenizer

```
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    load_in_4bit=True,
    max_seq_length=4096,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
```

### 3️⃣ Inference Function

```
def do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens=4096):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")

    generated_token_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0, top_p=0.95, top_k=64,
    )

    generated_text = tokenizer.decode(
        generated_token_ids[len(inputs['input_ids']):],
        skip_special_tokens=True,
    )

    import gc, torch
    del inputs
    del generated_token_ids
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text
```

### 4️⃣ Extract Medical Test Names

```
report_image = "(https://raw.githubusercontent.com/DeepLumiere/MedOCR/blob/main/labimage.png"

list_tests_messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": report_image},
        {"type": "text", "text": "From this medical report image, please list all the distinct medical test names you can identify. Present them as a comma-separated list."}
    ]
}]

test_names = do_gemma_3n_inference(model, tokenizer, list_tests_messages, max_new_tokens=128)
print(test_names)
```

### 5️⃣ Analyze Specific Test

```
test_name = "MCHC"
detailed_messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": report_image},
        {"type": "text", "text": f"{test_name}. Find value, review whether it's in normal range, if not give potential reasons."}
    ]
}]
detailed_output = do_gemma_3n_inference(model, tokenizer, detailed_messages, max_new_tokens=128)
print(detailed_output)
```

### 6️⃣ Save Fine-Tuned Model & Tokenizer

```
model.save_pretrained("gemma-3n")
tokenizer.save_pretrained("gemma-3n")
```

---

## 🤝 Contributing

We welcome contributions! Please open issues or pull requests for improvements and new features.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Kaggle Gemma-3N Challenge Team  
- Gemma-3N and Unsloth FastModels communities  

---

**Empowering faster, smarter personal health insights with AI—anytime, anywhere!** 🚀🩺
```

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/81604858/9fe4535a-fbcd-43d4-ab85-e7c4beb02603/gemma-3n-submission-3.ipynb
