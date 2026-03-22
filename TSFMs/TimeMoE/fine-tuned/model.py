import torch
import numpy as np
from typing import List, Union
from transformers import AutoModelForCausalLM
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeMoEModel:
    def __init__(
            self,
            model_path: str = "Maple728/TimeMoE-200M",
            device: str = "cuda",
    ):
        """
        Time-MoE Model Wrapper
        
        Args:
            model_path: Path to the model (pre-trained or fine-tuned)
            device: 'cuda' or 'cpu' (default: cuda, terminates if unavailable)
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA is not available. Terminating process.")
            sys.exit(1)
        
        self.model_path = model_path
        self.device = device
        
        logger.info(f"Loading Time-MoE model from {model_path}")
        logger.info(f"Device: {device}")
        
        # Try Flash Attention 2 first
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else "cpu",
                attn_implementation='flash_attention_2',
                trust_remote_code=True,
            )
            logger.info("✅ Model loaded with Flash Attention 2")
        except Exception as e:
            # Use standard implementation when Flash Attention is unavailable
            logger.warning(f"Flash Attention 2 not available: {str(e)[:100]}...")
            logger.info("Loading model with standard attention...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if device == "cuda" else "cpu",
                    trust_remote_code=True,
                )
                logger.info("✅ Model loaded with standard attention")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                sys.exit(1)
        
        self.model.eval()
        logger.info(f"Model ready on {device}")

    def normalize(self, seqs: torch.Tensor) -> tuple:
        """
        Normalize time series
        
        Args:
            seqs: [batch_size, seq_len]
        
        Returns:
            (normed_seqs, mean, std)
        """
        mean = seqs.mean(dim=-1, keepdim=True)
        std = seqs.std(dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        normed_seqs = (seqs - mean) / std
        return normed_seqs, mean, std

    def denormalize(self, normed_seqs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize"""
        return normed_seqs * std + mean

    @torch.no_grad()
    def predict(
            self,
            context: Union[List[np.ndarray], np.ndarray, torch.Tensor],
            prediction_length: int,
    ) -> np.ndarray:
        """
        Predict future values
        
        Args:
            context: Input sequence
                - List[np.ndarray]: Multiple sequences, each with shape [seq_len]
                - np.ndarray: [batch_size, seq_len] or [seq_len]
                - torch.Tensor: [batch_size, seq_len]
            prediction_length: Prediction length
        
        Returns:
            predictions: [batch_size, prediction_length]
        """
        if isinstance(context, list):
            context = np.array([np.asarray(c).flatten() for c in context])
        
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float()
        
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        context = context.to(self.model.device)
        
        normed_context, mean, std = self.normalize(context)
        
        output = self.model.generate(
            normed_context,
            max_new_tokens=prediction_length
        )
        
        normed_predictions = output[:, -prediction_length:]
        
        predictions = self.denormalize(normed_predictions, mean, std)
        
        return predictions.cpu().numpy()

    @torch.no_grad()
    def predict_batch(
            self,
            contexts: List[np.ndarray],
            prediction_length: int,
            batch_size: int = 128
    ) -> np.ndarray:
        """
        Batch prediction (processing large number of samples)
        
        Args:
            contexts: List of input sequence
            prediction_length: Prediction length
            batch_size: Batch size
        
        Returns:
            predictions: [num_samples, prediction_length]
        """
        all_predictions = []
        
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_preds = self.predict(batch_contexts, prediction_length)
            all_predictions.append(batch_preds)
        
        return np.vstack(all_predictions)