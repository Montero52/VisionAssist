from torch import nn
from src.main.encode.vit import ViT
from src.main.decode.caption_decoder import TransformerDecoder
import torch
import torch.nn.functional as F
import logging
import config
from .openvino_engine import OpenVINOEngine

logger = logging.getLogger("ViT-Transformer")

class ViT_Transformer(nn.Module):
    def __init__(
        self,
        vit_config: dict,
        trans_cfg: dict,
        vocab_size: int = None, # Allow None to infer from config
        max_len: int = 32, 
        use_openvino: bool = False
    ):
        super().__init__()
        
        self.use_openvino = use_openvino
        if self.use_openvino:
            logger.info("Initializing ViT-Transformer in OpenVINO Mode...")
            self.ov_engine = OpenVINOEngine()
        else:
            logger.info("Initializing ViT-Transformer in PyTorch Mode...")
        
        # [FIX 1] Priority: passed vocab_size -> config -> default T5
        self.vocab_size = vocab_size if vocab_size is not None else trans_cfg.get('vocab_size', 32128)
        
        logger.info(f"Initializing ViT-Transformer Decoder with Vocab Size = {self.vocab_size}")
        
        # 1. Encoder: Vision Transformer (ViT)
        self.encoder = ViT(
            image_size=vit_config.get("image_size", 224),
            patch_size=vit_config.get("patch_size", 16),
            in_channels=vit_config.get("in_channels", 3),
            embed_dim=vit_config.get("embed_dim", 768),
            depth=vit_config.get("depth", 12),
            num_heads=vit_config.get("num_heads", 12),
            mlp_ratio=vit_config.get("mlp_ratio", 4.0),
            dropout=vit_config.get("dropout", 0.0),
            pretrained=not self.use_openvino, # Don't load weights if using OV
        )

        # 2. Decoder: Custom Transformer
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size, # Use self.vocab_size (already handled None)
            dim=trans_cfg.get("dim", 512),
            num_heads=trans_cfg.get("num_heads", 8),
            num_layers=trans_cfg.get("num_layers", 6),
            ff_dim=trans_cfg.get("ff_dim", 2048),
            dropout=trans_cfg.get("dropout", 0.0),
            max_len=trans_cfg.get("max_len", max_len)
        )

        # 3. Projection Layer (Connect Encoder -> Decoder)
        vit_dim = vit_config.get("embed_dim", 768)
        trans_dim = trans_cfg.get("dim", 512)
        
        if vit_dim != trans_dim:
            self.proj = nn.Linear(vit_dim, trans_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            images: [Batch, 3, H, W] tensor
            input_ids: [Batch, Seq_Len] tensor (with Start Token)
            padding_mask: [Batch, Seq_Len] (1 for real token, 0 for pad)
        """
        if self.use_openvino:
            logger.warning("Forward pass not recommended in OpenVINO mode. Use beam_search().")
            
        features = self.encoder(images) 
        encoder_out = self.proj(features)
      
        # Decoder Masking Logic
        T = input_ids.size(1)
        device = input_ids.device

        # 1. Causal Mask (Look-ahead mask) - Lower triangular matrix
        tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]

        # 2. Padding Mask (Ignore pad tokens)
        if padding_mask is not None:
            # padding_mask: [Batch, T] -> [Batch, 1, 1, T]
            expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            # Combine: Keep positions that are (Past + Present) AND (Not Pad)
            tgt_mask = tgt_mask * expanded_padding_mask

        # 3. Convert to float for Softmax (-inf to zero out attention)
        attention_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
        attention_mask = attention_mask.masked_fill(tgt_mask == 0, float('-inf'))

        logits = self.decoder(input_ids, encoder_out, attention_mask)
        return logits

    @torch.no_grad()
    def beam_search(self, image: torch.Tensor, tokenizer, beam_size=3, max_len=30, device="cpu", alpha=0.7, no_repeat_ngram_size=2, repetition_penalty=1.0):
        """
        Improved Beam Search with N-gram Blocking to prevent repetitive output.
        Automatically switches between PyTorch and OpenVINO modes.
        """
        if self.use_openvino:
            return self.ov_engine.beam_search(
                image_tensor=image,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_len=max_len,
                alpha=alpha,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty
            )
            
        self.eval()
        
        # 1. Image Encoding
        features = self.encoder(image)
        encoder_out = self.proj(features)
        
        # Expand Encoder Output for each Beam: [Batch*Beam, Seq, Dim]
        encoder_out = encoder_out.expand(beam_size, -1, -1)

        # Tokenizer ID setup (T5)
        start_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # sequences: List containing [list_token_ids, cumulative_score]
        sequences = [[list(), 0.0]] 
        
        for step in range(max_len):
            all_candidates = []
            num_current_beams = 1 if step == 0 else len(sequences)
            
            # --- Prepare Batch Input for Decoder ---
            batch_seqs = []
            max_curr_len = max([len(seq[0]) for seq in sequences]) + 1 
            
            for seq in sequences:
                tokens = seq[0]
                full_seq = [start_token_id] + tokens
                num_pads = max_curr_len - len(full_seq)
                full_seq = full_seq + [pad_token_id] * num_pads
                batch_seqs.append(full_seq)
            
            input_ids = torch.tensor(batch_seqs, device=device) 
            curr_encoder_out = encoder_out[:num_current_beams]
            
            # --- Create Masks ---
            T = input_ids.size(1)
            tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            pad_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1) 
            pad_mask[:, :, :, 0] = 1 # Fix for T5 start token offset
            
            tgt_mask = tgt_mask * pad_mask
            attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
            
            # Decoder Forward
            logits = self.decoder(input_ids, curr_encoder_out, attention_mask)
            
            # Calculate Log Softmax
            next_token_probs_batch = F.log_softmax(logits, dim=-1)
            
            # Iterate through Beams
            for i in range(num_current_beams):
                seq, score = sequences[i]
                
                # Check if beam has terminated
                if len(seq) > 0 and seq[-1] == eos_token_id:
                    all_candidates.append([seq, score])
                    continue
                
                curr_idx = len(seq) 
                # Get current step probabilities (Clone to avoid modifying other beams)
                next_token_probs = next_token_probs_batch[i, curr_idx, :].clone()
                
                # =========================================================
                # REPETITION HANDLING
                # =========================================================
                
                # 1. Repetition Penalty
                if repetition_penalty > 1.0:
                    for token_id in set(seq):
                        if next_token_probs[token_id] < 0:
                            next_token_probs[token_id] /= repetition_penalty
                        else:
                            next_token_probs[token_id] *= repetition_penalty

                # 2. N-gram Blocking
                if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    
                    for idx in range(len(seq) - no_repeat_ngram_size + 1):
                        past_gram = tuple(seq[idx : idx + no_repeat_ngram_size - 1])
                        if past_gram == prefix:
                            banned_token = seq[idx + no_repeat_ngram_size - 1]
                            next_token_probs[banned_token] = -float('inf')

                # =========================================================
                
                # Select Top K candidates
                top_k_probs, top_k_ids = next_token_probs.topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = seq + [top_k_ids[j].item()]
                    new_score = score + top_k_probs[j].item()
                    all_candidates.append([new_seq, new_score])

            # Sort and select best beams (Length penalty included)
            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) + 1) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            
            if len(sequences[0][0]) > 0 and sequences[0][0][-1] == eos_token_id:
                break

        best_seq = sequences[0][0]
        caption = tokenizer.decode(best_seq, skip_special_tokens=True)
        return caption
   
    def apply_no_repeat_ngram(self, logits: torch.Tensor, history_tokens: list, ngram_size=2) -> torch.Tensor:
        """
        N-gram Blocking implementation.
        - logits: Predicted scores for current step [vocab_size]
        - history_tokens: Generated token ids
        - ngram_size: Forbidden repetitive phrase size
        """
        if len(history_tokens) < ngram_size - 1:
            return logits

        prefix = tuple(history_tokens[-(ngram_size - 1):])

        banned_indices = set()
        for i in range(len(history_tokens) - ngram_size + 1):
            past_gram = tuple(history_tokens[i : i + ngram_size - 1])
            if past_gram == prefix:
                banned_token = history_tokens[i + ngram_size - 1]
                banned_indices.add(banned_token)

        for idx in banned_indices:
            logits[idx] = -float('inf')

        return logits
