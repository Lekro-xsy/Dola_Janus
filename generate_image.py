import torch
from transformers import AutoModelForCausalLM
import json

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
import random

def main():
    # Load prompts
    prompt_file = "/z_data/syxin/janus/prompt.json"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = data['prompts']
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")
    
    # Set random seed for reproducibility (same as run_dola.py)
    # Fix seeds across libraries for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # specify the path to the model
    model_path = "/z_data/syxin/janus/Janus-Pro-7B"
    print("Loading Janus-Pro-7B model...")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print("Model loaded successfully!")
    
    # Create save directory
    save_dir = '/z_data/syxin/janus/normal'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Images will be saved to: {save_dir}")
    
    # Process each prompt
    for idx, prompt_text in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Processing prompt {idx}/{len(prompts)}")
        print(f"Prompt: {prompt_text}")
        print('='*60)
        
        conversation = [
            {
                "role": "User",
                "content": prompt_text,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        
        # Generate image
        save_path = os.path.join(save_dir, f"{idx}.jpg")
        generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            save_path=save_path
        )
        print(f"Saved: {save_path}")
    
    print(f"\n🎉 Batch processing completed! Generated {len(prompts)} images in {save_dir}")


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    save_path: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        if i % 100 == 0:
            print(f"Progress: {i}/{image_token_num_per_image}")
            
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    print("Decoding images...")
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # Save image
    PIL.Image.fromarray(visual_img[0]).save(save_path)


if __name__ == "__main__":
    main()
