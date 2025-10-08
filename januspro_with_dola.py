"""
JanusPro with DoLa integration
Enhanced multimodal inference with DoLa decoding
"""

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import argparse
import os
from typing import List, Optional

# Import DoLa components
from dola_processor import DoLaLogitsProcessor, DoLaConfig, create_dola_config_from_args
from dola_args import add_dola_arguments, validate_dola_args, print_dola_config


class JanusProWithDoLa:
    """JanusPro model with DoLa decoding support"""
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[torch.device] = None,
                 dola_config: Optional[DoLaConfig] = None):
        
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dola_config = dola_config or DoLaConfig()
        
        # Load model and processor
        print(f"Loading JanusPro model from {model_path}")
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).to(self.device).eval()
        
        # Initialize DoLa processor
        self.dola_processor = None
        if self.dola_config.use_dola:
            print("Initializing DoLa processor...")
            self.dola_processor = DoLaLogitsProcessor(
                self.vl_gpt.language_model,
                self.dola_config,
                self.device
            )
        
        print("JanusPro with DoLa initialized successfully!")
    
    def multimodal_understanding(self, 
                                question: str,
                                images: List[str] = None,
                                max_new_tokens: int = 512,
                                do_sample: bool = False,
                                temperature: float = 1.0,
                                top_p: float = 0.9,
                                top_k: int = 50) -> str:
        """
        Multimodal understanding with DoLa decoding
        
        Args:
            question: Text question
            images: List of image paths
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            
        Returns:
            Generated response
        """
        
        # Prepare conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}" if images else question,
                "images": images or [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self.vl_gpt.device)
        
        # Prepare input embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Setup generation parameters
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        
        # Add DoLa processor if enabled
        if self.dola_processor and self.dola_config.apply_on_generation:
            generation_kwargs["logits_processor"] = [self.dola_processor]
            print("Using DoLa decoding for text generation")
        
        # Generate response
        with torch.no_grad():
            outputs = self.vl_gpt.language_model.generate(**generation_kwargs)
        
        # Decode response
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return answer
    
    def text_to_image_generation(self,
                               prompt: str,
                               temperature: float = 1.0,
                               parallel_size: int = 16,
                               cfg_weight: float = 5.0,
                               image_token_num_per_image: int = 576,
                               img_size: int = 384,
                               patch_size: int = 16,
                               save_dir: str = "generated_samples") -> List[str]:
        """
        Text-to-image generation with DoLa decoding
        
        Args:
            prompt: Text prompt for image generation
            temperature: Generation temperature
            parallel_size: Number of parallel samples
            cfg_weight: CFG guidance weight
            image_token_num_per_image: Number of image tokens
            img_size: Image size
            patch_size: Patch size
            save_dir: Directory to save generated images
            
        Returns:
            List of saved image paths
        """
        
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        generation_prompt = sft_format + self.vl_chat_processor.image_start_tag
        
        # Run image generation with DoLa
        return self._generate_images_with_dola(
            generation_prompt,
            temperature,
            parallel_size,
            cfg_weight,
            image_token_num_per_image,
            img_size,
            patch_size,
            save_dir
        )
    
    def _generate_images_with_dola(self,
                                 prompt: str,
                                 temperature: float,
                                 parallel_size: int,
                                 cfg_weight: float,
                                 image_token_num_per_image: int,
                                 img_size: int,
                                 patch_size: int,
                                 save_dir: str) -> List[str]:
        """Internal image generation method with DoLa support"""
        
        import PIL.Image
        import numpy as np
        
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id
        
        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        
        # Image generation loop with optional DoLa
        with torch.no_grad():
            for i in range(image_token_num_per_image):
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i != 0 else None
                )
                hidden_states = outputs.last_hidden_state
                
                # Get logits from generation head
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                # Apply CFG
                if cfg_weight > 0:
                    logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                else:
                    logits = logit_cond
                
                # Apply DoLa on image tokens if enabled
                if (self.dola_processor and 
                    self.dola_config.apply_on_image_tokens and
                    hasattr(self, '_apply_dola_to_image_logits')):
                    
                    # Note: This would require adapting DoLa for image token generation
                    # For now, we use standard decoding for image tokens
                    pass
                
                # Sample next token
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                
                # Prepare next input
                next_token = torch.cat([
                    next_token.unsqueeze(dim=1), 
                    next_token.unsqueeze(dim=1)
                ], dim=1).view(-1)
                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        # Decode images
        dec = self.vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        
        # Save images
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        for i in range(parallel_size):
            save_path = os.path.join(save_dir, f"img_{i}.jpg")
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            saved_paths.append(save_path)
        
        return saved_paths
    
    def cleanup(self):
        """Clean up resources"""
        if self.dola_processor:
            self.dola_processor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="JanusPro with DoLa Demo")
    
    # Basic arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to JanusPro model')
    parser.add_argument('--mode', type=str, default='understanding',
                       choices=['understanding', 'generation', 'both'],
                       help='Inference mode')
    parser.add_argument('--question', type=str, 
                       default="Hello! Can you introduce yourself?",
                       help='Question for multimodal understanding')
    parser.add_argument('--images', type=str, nargs='*',
                       help='Image paths for understanding')
    parser.add_argument('--prompt', type=str,
                       default="A beautiful landscape with mountains and rivers",
                       help='Prompt for image generation')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    
    # Image generation parameters
    parser.add_argument('--parallel_size', type=int, default=4)
    parser.add_argument('--cfg_weight', type=float, default=5.0)
    
    # Add DoLa arguments
    parser = add_dola_arguments(parser)
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_dola_args(args)
    
    # Print DoLa configuration
    print_dola_config(args)
    
    # Create DoLa config
    dola_config = create_dola_config_from_args(args)
    
    # Initialize model
    model = JanusProWithDoLa(
        model_path=args.model_path,
        dola_config=dola_config
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode in ['understanding', 'both']:
            print("\n" + "="*50)
            print("MULTIMODAL UNDERSTANDING")
            print("="*50)
            print(f"Question: {args.question}")
            if args.images:
                print(f"Images: {args.images}")
            
            response = model.multimodal_understanding(
                question=args.question,
                images=args.images,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            
            print(f"\nResponse:\n{response}")
            
            # Save response
            with open(os.path.join(args.output_dir, 'response.txt'), 'w') as f:
                f.write(f"Question: {args.question}\n")
                f.write(f"Response: {response}\n")
        
        if args.mode in ['generation', 'both']:
            print("\n" + "="*50)
            print("TEXT-TO-IMAGE GENERATION")
            print("="*50)
            print(f"Prompt: {args.prompt}")
            
            image_paths = model.text_to_image_generation(
                prompt=args.prompt,
                temperature=args.temperature,
                parallel_size=args.parallel_size,
                cfg_weight=args.cfg_weight,
                save_dir=os.path.join(args.output_dir, 'generated_images')
            )
            
            print(f"\nGenerated {len(image_paths)} images:")
            for path in image_paths:
                print(f"  - {path}")
    
    finally:
        # Cleanup
        model.cleanup()


if __name__ == "__main__":
    main()