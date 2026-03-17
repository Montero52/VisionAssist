from torch import nn
from src.main.encode.vit import ViT
from src.main.decode.caption_decoder import TransformerDecoder
import torch
import torch.nn.functional as F

class ViT_Transformer(nn.Module):
    def __init__(
        self,
        vit_config: dict,
        trans_cfg: dict,
        vocab_size: int = None, # Cho phép None để tự lấy từ config
        max_len: int = 32, 
    ):
        super().__init__()
        
        # [FIX 1] Ưu tiên lấy vocab_size truyền vào -> rồi đến config -> cuối cùng là default T5
        self.vocab_size = vocab_size if vocab_size is not None else trans_cfg.get('vocab_size', 32128)
        
        print(f"DEBUG MODEL: Initializing Decoder with Vocab Size = {self.vocab_size}")
        
        # 1. Encoder: ViT
        self.encoder = ViT(
            image_size=vit_config.get("image_size", 224),
            patch_size=vit_config.get("patch_size", 16),
            in_channels=vit_config.get("in_channels", 3),
            embed_dim=vit_config.get("embed_dim", 768),
            depth=vit_config.get("depth", 12),
            num_heads=vit_config.get("num_heads", 12),
            mlp_ratio=vit_config.get("mlp_ratio", 4.0),
            dropout=vit_config.get("dropout", 0.0),
            pretrained=True,
        )

        # 2. Decoder: Transformer
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size, # <--- [QUAN TRỌNG] Phải dùng self.vocab_size (đã xử lý None)
            dim=trans_cfg.get("dim", 512),
            num_heads=trans_cfg.get("num_heads", 8),
            num_layers=trans_cfg.get("num_layers", 6),
            ff_dim=trans_cfg.get("ff_dim", 2048),
            dropout=trans_cfg.get("dropout", 0.0),
            max_len=trans_cfg.get("max_len", max_len)
        )

        # 3. Projection Layer (Kết nối Encoder -> Decoder)
        vit_dim = vit_config.get("embed_dim", 768)
        trans_dim = trans_cfg.get("dim", 512)
        
        if vit_dim != trans_dim:
            self.proj = nn.Linear(vit_dim, trans_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, images, input_ids, padding_mask=None):
        """
        images: [Batch, 3, H, W]
        input_ids: [Batch, Seq_Len] (đã thêm Start Token)
        padding_mask: [Batch, Seq_Len] (1 là token thật, 0 là pad)
        """
        features = self.encoder(images) 
        encoder_out = self.proj(features)
      
        # Decoder Masking Logic
        T = input_ids.size(1)
        device = input_ids.device

        # 1. Causal Mask (Che tương lai) - Ma trận tam giác dưới
        tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]

        # 2. Padding Mask (Che token rác)
        if padding_mask is not None:
            # padding_mask: [Batch, T] -> [Batch, 1, 1, T]
            expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            # Kết hợp: Chỉ giữ lại vị trí (Quá khứ + Hiện tại) VÀ (Không phải Pad)
            tgt_mask = tgt_mask * expanded_padding_mask

        # 3. Convert to float for Softmax (-inf để triệt tiêu attention)
        attention_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
        attention_mask = attention_mask.masked_fill(tgt_mask == 0, float('-inf'))

        logits = self.decoder(input_ids, encoder_out, attention_mask)
        return logits

    @torch.no_grad()
    def beam_search(self, image, tokenizer, beam_size=3, max_len=30, device="cpu", alpha=0.7, no_repeat_ngram_size=2, repetition_penalty=1.2):
        """
        Beam Search (Final Version): Tích hợp N-gram Blocking & Repetition Penalty chuẩn xác.
        """
        self.eval()
        
        # 1. ENCODER: Trích xuất đặc trưng ảnh
        features = self.encoder(image)
        encoder_out = self.proj(features)
        # Mở rộng cho từng beam: [Batch*Beam, Seq, Dim]
        encoder_out = encoder_out.expand(beam_size, -1, -1)

        # 2. CHUẨN BỊ TOKEN
        start_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # sequences: List chứa [list_token_ids, cumulative_score]
        sequences = [[list(), 0.0]] 
        
        # 3. VÒNG LẶP DECODER (Sinh từng từ)
        for step in range(max_len):
            all_candidates = []
            num_current_beams = 1 if step == 0 else len(sequences)
            
            # --- A. Chuẩn bị Batch Input ---
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
            
            # --- B. Tạo Mask & Forward ---
            T = input_ids.size(1)
            tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            pad_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1) 
            pad_mask[:, :, :, 0] = 1 
            tgt_mask = tgt_mask * pad_mask
            attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
            
            logits = self.decoder(input_ids, curr_encoder_out, attention_mask)
            
            # Lấy Log Softmax (Giá trị là số ÂM)
            next_token_probs_batch = F.log_softmax(logits, dim=-1)
            
            # --- C. Xử lý từng Beam ---
            for i in range(num_current_beams):
                seq, score = sequences[i]
                
                # Nếu câu đã kết thúc (gặp EOS)
                if len(seq) > 0 and seq[-1] == eos_token_id:
                    all_candidates.append([seq, score])
                    continue
                
                curr_idx = len(seq) 
                next_token_probs = next_token_probs_batch[i, curr_idx, :].clone()
                
                # =========================================================
                # LOGIC CHẶN LẶP (ANTI-REPETITION)
                # =========================================================
                
                # 1. Repetition Penalty (Phạt lặp từ đơn)
                # Logic: Log_prob là số âm. Nhân với số > 1 sẽ làm nó âm hơn (giảm điểm).
                if repetition_penalty > 1.0:
                    for token_id in set(seq):
                        if next_token_probs[token_id] < 0:
                            next_token_probs[token_id] *= repetition_penalty
                        else:
                            next_token_probs[token_id] /= repetition_penalty

                # 2. N-gram Blocking (Chặn lặp cụm từ)
                if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    
                    for idx in range(len(seq) - no_repeat_ngram_size + 1):
                        past_gram = tuple(seq[idx : idx + no_repeat_ngram_size - 1])
                        if past_gram == prefix:
                            banned_token = seq[idx + no_repeat_ngram_size - 1]
                            next_token_probs[banned_token] = -float('inf') # Cấm tuyệt đối

                # =========================================================
                
                # --- D. Chọn Top-K ---
                top_k_probs, top_k_ids = next_token_probs.topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = seq + [top_k_ids[j].item()]
                    new_score = score + top_k_probs[j].item()
                    all_candidates.append([new_seq, new_score])

            # --- E. Sắp xếp & Chọn lọc ---
            # Length Normalization: Chia điểm cho độ dài để công bằng với câu dài
            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) + 1) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            
            # Dừng sớm nếu beam tốt nhất đã xong
            if len(sequences[0][0]) > 0 and sequences[0][0][-1] == eos_token_id:
                break

        # 4. GIẢI MÃ KẾT QUẢ
        best_seq = sequences[0][0]
        caption = tokenizer.decode(best_seq, skip_special_tokens=True)
        return caption