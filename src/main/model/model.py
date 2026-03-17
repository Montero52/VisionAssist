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
    def beam_search(self, image, tokenizer, beam_size=3, max_len=30, device="cpu", alpha=0.7, no_repeat_ngram_size=2, repetition_penalty=1.0):
        """
        Beam Search Cải tiến: Tích hợp N-gram Blocking để chống lặp từ.
        Args:
            - no_repeat_ngram_size (int): Kích thước cụm từ cấm lặp lại (VD: 2 cấm "box of" ... "box of").
            - repetition_penalty (float): > 1.0 sẽ phạt các từ đã xuất hiện (giảm xác suất của chúng).
        """
        self.eval()
        
        # 1. Encode ảnh
        features = self.encoder(image)
        encoder_out = self.proj(features)
        
        # Mở rộng Encoder Output cho từng Beam: [Batch*Beam, Seq, Dim]
        encoder_out = encoder_out.expand(beam_size, -1, -1)

        # T5 Tokenizer IDs
        start_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # sequences: List chứa [list_token_ids, cumulative_score]
        sequences = [[list(), 0.0]] 
        
        for step in range(max_len):
            all_candidates = []
            num_current_beams = 1 if step == 0 else len(sequences)
            
            # --- Chuẩn bị Batch Input cho Decoder ---
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
            
            # --- Tạo Mask ---
            T = input_ids.size(1)
            tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            pad_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1) 
            pad_mask[:, :, :, 0] = 1 # Fix cho T5 start token
            
            tgt_mask = tgt_mask * pad_mask
            attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
            
            # Forward Decoder
            logits = self.decoder(input_ids, curr_encoder_out, attention_mask)
            
            # Lấy Log Softmax
            next_token_probs_batch = F.log_softmax(logits, dim=-1)
            
            # Duyệt qua từng Beam
            for i in range(num_current_beams):
                seq, score = sequences[i]
                
                # Nếu beam này đã kết thúc
                if len(seq) > 0 and seq[-1] == eos_token_id:
                    all_candidates.append([seq, score])
                    continue
                
                curr_idx = len(seq) 
                # Lấy xác suất bước hiện tại (Clone để không ảnh hưởng beam khác)
                next_token_probs = next_token_probs_batch[i, curr_idx, :].clone()
                
                # =========================================================
                # 🛠️ [CODE MỚI] XỬ LÝ LẶP TỪ (REPETITION HANDLING) 🛠️
                # =========================================================
                
                # 1. Repetition Penalty (Phạt nhẹ từ đã xuất hiện)
                if repetition_penalty > 1.0:
                    for token_id in set(seq):
                        # Nếu log_prob < 0, chia cho penalty (>1) sẽ làm nó nhỏ hơn (âm hơn)
                        # Nếu log_prob > 0 (hiếm), nhân với penalty
                        if next_token_probs[token_id] < 0:
                            next_token_probs[token_id] /= repetition_penalty
                        else:
                            next_token_probs[token_id] *= repetition_penalty

                # 2. N-gram Blocking (Cấm tuyệt đối lặp cụm từ)
                if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                    # Lấy (N-1) từ cuối cùng làm tiền tố
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    
                    # Quét lại quá khứ để xem tiền tố này từng xuất hiện ở đâu
                    for idx in range(len(seq) - no_repeat_ngram_size + 1):
                        # Lấy cụm N-1 từ trong quá khứ
                        past_gram = tuple(seq[idx : idx + no_repeat_ngram_size - 1])
                        
                        if past_gram == prefix:
                            # Tìm thấy lặp! Từ tiếp theo trong quá khứ là từ cấm
                            banned_token = seq[idx + no_repeat_ngram_size - 1]
                            # Gán điểm Âm Vô Cùng để xác suất về 0
                            next_token_probs[banned_token] = -float('inf')

                # =========================================================
                
                # Chọn Top K (Sau khi đã phạt/cấm từ lặp)
                top_k_probs, top_k_ids = next_token_probs.topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = seq + [top_k_ids[j].item()]
                    new_score = score + top_k_probs[j].item()
                    all_candidates.append([new_seq, new_score])

            # Chọn beam tốt nhất
            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) + 1) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            
            if len(sequences[0][0]) > 0 and sequences[0][0][-1] == eos_token_id:
                break

        best_seq = sequences[0][0]
        caption = tokenizer.decode(best_seq, skip_special_tokens=True)
        return caption
  
    def apply_no_repeat_ngram(logits, history_tokens, ngram_size=2):
      """
      Hàm chặn lặp từ (N-gram Blocking).
      - logits: Điểm số dự đoán của bước hiện tại [vocab_size]
      - history_tokens: List các token id đã sinh ra trước đó
      - ngram_size: Kích thước cụm từ muốn chặn (VD: 2 nghĩa là cấm lặp lại cặp từ liên tiếp)
      """
      if len(history_tokens) < ngram_size - 1:
          return logits # Chưa đủ dài để check, bỏ qua

      # Lấy (N-1) từ cuối cùng vừa sinh ra làm "tiền tố"
      prefix = tuple(history_tokens[-(ngram_size - 1):])

      # Quét lại toàn bộ lịch sử để xem "tiền tố" này từng xuất hiện ở đâu
      # Và từ gì đã đi theo sau nó?
      banned_indices = set()
      for i in range(len(history_tokens) - ngram_size + 1):
          # Kiểm tra đoạn token trong quá khứ
          past_gram = tuple(history_tokens[i : i + ngram_size - 1])
          
          # Nếu đoạn quá khứ giống hệt đoạn hiện tại
          if past_gram == prefix:
              # Thì cái từ đi sau đoạn quá khứ đó là "từ cấm"
              banned_token = history_tokens[i + ngram_size - 1]
              banned_indices.add(banned_token)

      # Gán logit của các từ cấm thành âm vô cùng (để Softmax ra 0%)
      for idx in banned_indices:
          logits[idx] = -float('inf')

      return logits