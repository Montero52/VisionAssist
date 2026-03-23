import openvino.runtime as ov
import numpy as np
import torch
import logging
from typing import Dict, Union, Any
import config

logger = logging.getLogger("OpenVINOEngine")

class OpenVINOEngine:
    """
    Class to manage OpenVINO model loading and inference for Captioning and Depth Estimation.
    Optimized for Intel CPU and iGPU using 'AUTO' device.
    """
    def __init__(self):
        self.core = ov.Core()
        self.models: Dict[str, Any] = {}
        self.device = config.DEVICE
        
        # Load models if USE_OPENVINO is enabled
        if config.USE_OPENVINO:
            self._load_all_models()
        else:
            logger.info("OpenVINO is disabled in config. Skipping model loading.")

    def _load_all_models(self) -> None:
        """Loads and compiles encoder, decoder, and depth models."""
        try:
            # 1. Captioning Encoder
            logger.info(f"Loading Encoder from: {config.ENCODER_XML}")
            encoder_model = self.core.read_model(config.ENCODER_XML)
            self.models['encoder'] = self.core.compile_model(encoder_model, self.device)
            
            # 2. Captioning Decoder
            logger.info(f"Loading Decoder from: {config.DECODER_XML}")
            decoder_model = self.core.read_model(config.DECODER_XML)
            self.models['decoder'] = self.core.compile_model(decoder_model, self.device)
            
            # 3. Depth Estimator
            logger.info(f"Loading Depth Model from: {config.DEPTH_XML}")
            depth_model = self.core.read_model(config.DEPTH_XML)
            self.models['depth'] = self.core.compile_model(depth_model, self.device)
            
            logger.info("All OpenVINO models compiled successfully!")
        except Exception as e:
            logger.error(f"Failed to load OpenVINO models: {e}")
            raise e

    def infer_encoder(self, image_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Runs inference for the ViT Encoder.
        Args:
            image_tensor: Normalized image tensor [1, 3, 224, 224]
        Returns:
            Encoded features: [1, 197, 768]
        """
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.cpu().numpy()
            
        # Standardize input name if needed (OpenVINO models often use 'pixel_values' or 'images')
        # We assume the conversion script used 'images' as the input name for the encoder.
        result = self.models['encoder']([image_tensor])
        return list(result.values())[0]

    def infer_decoder(self, input_ids: np.ndarray, encoder_out: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Runs inference for the Transformer Decoder.
        Args:
            input_ids: [Batch, Seq_Len]
            encoder_out: [Batch, 197, 768]
            attention_mask: [Batch, 1, Seq_Len, Seq_Len]
        Returns:
            Logits: [Batch, Seq_Len, Vocab_Size]
        """
        # OpenVINO compiled_model can take a list or dict of inputs
        # Match input names from the XML
        inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_out,
            "attention_mask": attention_mask
        }
        result = self.models['decoder'](inputs)
        return list(result.values())[0]

    def infer_depth(self, pixel_values: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Runs inference for the Depth-Anything-V2 model.
        Args:
            pixel_values: Preprocessed image [1, 3, H, W]
        Returns:
            Depth Map: [1, H, W]
        """
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.cpu().numpy()
            
        result = self.models['depth']([pixel_values])
        return list(result.values())[0]

    def beam_search(self, image_tensor: Union[torch.Tensor, np.ndarray], tokenizer, beam_size=3, max_len=30, alpha=0.7, no_repeat_ngram_size=2, repetition_penalty=1.0) -> str:
        """
        Beam Search using OpenVINO compiled models.
        Optimized for performance on Intel CPU/iGPU.
        """
        # 1. Image Encoding (ViT Encoder)
        encoder_out = self.infer_encoder(image_tensor)
        
        # Expand Encoder Output for each Beam: [Beam, 197, 768]
        encoder_out = np.repeat(encoder_out, beam_size, axis=0)

        # Tokenizer Setup
        start_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # sequences: List containing [list_token_ids, cumulative_score]
        sequences = [[list(), 0.0]] 
        
        for step in range(max_len):
            all_candidates = []
            num_current_beams = 1 if step == 0 else len(sequences)
            
            # --- Prepare Batch Input for Decoder (Numpy-style) ---
            max_curr_len = max([len(seq[0]) for seq in sequences]) + 1 
            batch_seqs = []
            
            for seq in sequences:
                tokens = seq[0]
                full_seq = [start_token_id] + tokens
                num_pads = max_curr_len - len(full_seq)
                full_seq = full_seq + [pad_token_id] * num_pads
                batch_seqs.append(full_seq)
            
            input_ids = np.array(batch_seqs, dtype=np.int64)
            curr_encoder_out = encoder_out[:num_current_beams]
            
            # --- Create Masks (Numpy equivalents of model.py logic) ---
            # T = input_ids.shape[1]
            # tgt_mask: lower triangular matrix [1, 1, T, T]
            T = input_ids.shape[1]
            tgt_mask = np.tril(np.ones((T, T), dtype=np.float32)).reshape(1, 1, T, T)
            
            # pad_mask: [Batch, 1, 1, T]
            pad_mask = (input_ids != pad_token_id).astype(np.float32).reshape(num_current_beams, 1, 1, T)
            pad_mask[:, :, :, 0] = 1.0 # T5 start token offset fix
            
            tgt_mask = tgt_mask * pad_mask
            attention_mask = np.zeros_like(tgt_mask)
            attention_mask[tgt_mask == 0] = -1e9 # Mask with large negative value
            
            # --- Decoder Inference (OpenVINO) ---
            logits = self.infer_decoder(input_ids, curr_encoder_out, attention_mask)
            
            # --- Softmax & Search logic ---
            # Convert to log_softmax for scores (Numpy version)
            def log_softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return np.log(e_x / e_x.sum(axis=-1, keepdims=True))

            next_token_probs_batch = log_softmax(logits)
            
            for i in range(num_current_beams):
                seq, score = sequences[i]
                
                if len(seq) > 0 and seq[-1] == eos_token_id:
                    all_candidates.append([seq, score])
                    continue
                
                curr_idx = len(seq)
                next_token_probs = next_token_probs_batch[i, curr_idx, :].copy()
                
                # --- Repetition Handling (same as model.py) ---
                if repetition_penalty > 1.0:
                    for token_id in set(seq):
                        if next_token_probs[token_id] < 0:
                            next_token_probs[token_id] /= repetition_penalty
                        else:
                            next_token_probs[token_id] *= repetition_penalty

                if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    for idx in range(len(seq) - no_repeat_ngram_size + 1):
                        past_gram = tuple(seq[idx : idx + no_repeat_ngram_size - 1])
                        if past_gram == prefix:
                            banned_token = seq[idx + no_repeat_ngram_size - 1]
                            next_token_probs[banned_token] = -1e9

                # Select Top K
                top_k_ids = np.argsort(next_token_probs)[-beam_size:][::-1]
                top_k_probs = next_token_probs[top_k_ids]
                
                for j in range(beam_size):
                    new_seq = seq + [int(top_k_ids[j])]
                    new_score = score + top_k_probs[j]
                    all_candidates.append([new_seq, new_score])

            # Sorting & Pruning
            ordered = sorted(all_candidates, key=lambda x: x[1] / ((len(x[0]) + 1) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            
            if len(sequences[0][0]) > 0 and sequences[0][0][-1] == eos_token_id:
                break

        best_seq = sequences[0][0]
        return tokenizer.decode(best_seq, skip_special_tokens=True)
