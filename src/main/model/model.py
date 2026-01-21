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
        vocab_size: int,
        max_len: int = 32, # Tăng max_len để linh hoạt hơn
    ):
        super().__init__()

        # 1. Encoder: ViT
        self.encoder = ViT(
            image_size=vit_config.get("image_size", 224),
            patch_size=vit_config.get("patch_size", 32),
            in_channels=vit_config.get("in_channels", 3),
            embed_dim=vit_config.get("embed_dim", 768),
            depth=vit_config.get("depth", 12),
            num_heads=vit_config.get("num_heads", 12),
            mlp_ratio=vit_config.get("mlp_ratio", 4.0),
            dropout=vit_config.get("dropout", 0.1),
        )

        # 2. Decoder: Transfomer
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            dim=trans_cfg.get("dim", 512),
            num_heads=trans_cfg.get("num_heads", 8),
            num_layers=trans_cfg.get("num_layers", 6),
            ff_dim=trans_cfg.get("ff_dim", 2048),
            dropout=trans_cfg.get("dropout", 0.1),
            max_len=trans_cfg.get("max_len", max_len)
        )

        # 3. Projection Layer (Khớp kích thước giữa ViT và Decoder)
        vit_dim = vit_config.get("embed_dim", 768)
        trans_dim = trans_cfg.get("dim", 512)
        if vit_dim != trans_dim:
            self.proj = nn.Linear(vit_dim, trans_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, images, input_ids, padding_mask=None):
        # ... (Encoder giữ nguyên) ...
        features = self.encoder(images) 
        encoder_out = self.proj(features)

        # Decoder Masking Logic TINH CHỈNH
        T = input_ids.size(1)
        device = input_ids.device

        # 1. Look-ahead Mask (Tam giác dưới): [1, 1, T, T]
        # Che tương lai: Hàng i chỉ nhìn được cột 0..i
        tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # 2. Xử lý Padding Mask (Nếu có)
        if padding_mask is not None:
            # padding_mask gốc: [B, T] (1 là thật, 0 là pad)
            
            # Mở rộng chiều để khớp với [B, 1, T, T] (Broadcasting)
            # Ta muốn: Tại hàng i (từ đang sinh), ta không được nhìn vào cột j nếu cột j là Pad.
            # -> [B, 1, 1, T]
            expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            
            # Kết hợp: Vị trí hợp lệ phải THỎA MÃN CẢ 2 ĐIỀU KIỆN:
            # (1) Nằm trong quá khứ (tgt_mask == 1)
            # (2) Không phải là Padding (padding_mask == 1)
            # Phép nhân element-wise (1*1=1, 1*0=0) làm tốt việc này
            tgt_mask = tgt_mask * expanded_padding_mask

        # 3. Chuyển đổi sang định dạng float cho Softmax
        # Giá trị 1 -> 0.0 (giữ nguyên)
        # Giá trị 0 -> -inf (bị che)
        attention_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
        attention_mask = attention_mask.masked_fill(tgt_mask == 0, float('-inf'))

        # Gọi decoder
        logits = self.decoder(input_ids, encoder_out, attention_mask)
        return logits
    
    @torch.no_grad()
    def beam_search(self, image, tokenizer, beam_size=3, max_len=30, device="cpu"):
        """
        Thuật toán Beam Search để sinh caption chất lượng cao nhất (Dùng cho đánh giá/báo cáo).
        """
        self.eval()
        
        # 1. Encode ảnh (Chỉ làm 1 lần)
        features = self.encoder(image)
        encoder_out = self.proj(features) # [1, N, Dim]
        
        # Lặp lại encoder_out cho đủ số lượng beam (để xử lý song song)
        # [Beam, N, Dim]
        encoder_out = encoder_out.expand(beam_size, -1, -1)

        # 2. Khởi tạo
        # Start token
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        
        # Danh sách chứa các ứng viên (Candidate): mỗi phần tử là (điểm_số, chuỗi_token)
        # Điểm số khởi đầu là 0.0
        # Chuỗi token khởi đầu là [START]
        sequences = [[list(), 0.0]] 
        
        # Để bắt đầu, ta cần feed token đầu tiên vào model
        # Lúc đầu chỉ có 1 beam (beam gốc), sau bước 1 sẽ tách thành k beam
        input_ids = torch.tensor([[start_token_id]], device=device)
        
        # Duyệt qua từng bước (từng từ)
        for step in range(max_len):
            all_candidates = []
            
            # Với bước đầu tiên, ta chỉ chạy 1 lần (vì chưa tách beam). 
            # Từ bước 2 trở đi, ta chạy cho 'beam_size' ứng viên.
            num_current_beams = 1 if step == 0 else len(sequences)
            
            # Tạo input cho đợt này
            if step > 0:
                # Gom các chuỗi hiện tại thành tensor [Beam, T]
                input_ids = torch.tensor([ [start_token_id] + seq[0] for seq in sequences ], device=device)
            
            # --- Forward ---
            # Chỉ lấy đúng số lượng encoder_out tương ứng số beam hiện tại
            curr_encoder_out = encoder_out[:num_current_beams]
            
            # Tạo mask
            T = input_ids.size(1)
            tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
            attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
            
            # Chạy Decoder
            logits = self.decoder(input_ids, curr_encoder_out, attention_mask)
            
            # Lấy log xác suất của từ cuối cùng
            # [Beam, Vocab_Size]
            next_token_logits = logits[:, -1, :]
            next_token_probs = F.log_softmax(next_token_logits, dim=-1)

            # --- Mở rộng Beam ---
            # Duyệt qua từng beam hiện tại
            for i in range(num_current_beams):
                seq, score = sequences[i]
                
                # Nếu câu đã kết thúc (có EOS), giữ nguyên nó và không mở rộng nữa
                if len(seq) > 0 and seq[-1] == tokenizer.eos_token_id:
                    all_candidates.append([seq, score])
                    continue
                
                # Lấy top k từ có xác suất cao nhất cho beam này
                # top_k_probs: [Beam_size], top_k_ids: [Beam_size]
                top_k_probs, top_k_ids = next_token_probs[i].topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = seq + [top_k_ids[j].item()]
                    new_score = score + top_k_probs[j].item() # Cộng dồn log_prob
                    all_candidates.append([new_seq, new_score])

            # --- Chọn lọc (Pruning) ---
            # Sắp xếp các ứng viên theo điểm số giảm dần (lớn nhất đứng đầu)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            
            # Chỉ giữ lại k ứng viên tốt nhất
            sequences = ordered[:beam_size]
            
            # Kiểm tra dừng sớm: Nếu ứng viên tốt nhất đã có EOS thì dừng luôn (cho nhanh)
            # (Hoặc bạn có thể để chạy hết max_len cho chắc)
            if len(sequences[0][0]) > 0 and sequences[0][0][-1] == tokenizer.eos_token_id:
                break

        # 3. Trả về câu tốt nhất (Câu đầu tiên trong list)
        best_seq = sequences[0][0]
        caption = tokenizer.decode(best_seq, skip_special_tokens=True)
        return caption
