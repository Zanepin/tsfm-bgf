import torch
import numpy as np
import pandas as pd
from pathlib import Path
from chronos import BaseChronosPipeline, Chronos2Pipeline
from typing import List, Union, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Chronos2Model:
    """Chronos-2 model wrapper class (using official API, supports Full and LoRA fine-tuning)"""
    
    def __init__(
            self,
            model_path: str = None,
            device: str = "cuda"
    ):
        """
        Initialize model
        
        Args:
            model_path: Path to the model, if None the pre-trained model will be used
            device: Device (default: cuda)
        """
        self.device = device
        self.pipeline: Optional[Chronos2Pipeline] = None
        
        if model_path:
            self.load_finetuned_model(model_path)
        else:
            self.load_pretrained_model()

    def load_pretrained_model(self, model_name: str = "amazon/chronos-2"):
        """Load pre-trained Chronos-2 model"""
        try:
            logger.info(f"Loading pre-trained Chronos-2 model: {model_name}")
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
            )
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            raise

    def load_finetuned_model(self, model_path: str):
        """
        Load fine-tuned Chronos-2 model
        
        Automatically detects whether it is Full fine-tuning or LoRA fine-tuning:
        - If adapter_config.json exists -> LoRA adapter
        - If config.json exists -> Full model
        """
        try:
            model_path = Path(model_path)
            logger.info(f"Loading fine-tuned Chronos-2 model from: {model_path}")
            
            # ✅ Detect if it is a LoRA adapter or a full model
            adapter_config_path = model_path / "adapter_config.json"
            config_path = model_path / "config.json"
            
            if adapter_config_path.exists():
                # LoRA adapter mode
                logger.info("Detected LoRA adapter, loading base model + adapter")
                self._load_lora_model(model_path)
            elif config_path.exists():
                # Full model mode
                logger.info("Detected full fine-tuned model, loading directly")
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    str(model_path),
                    device_map=self.device,
                )
            else:
                raise ValueError(
                    f"Invalid model path: {model_path}\n"
                    f"Expected either adapter_config.json (LoRA) or config.json (Full)"
                )
            
            logger.info(f"Successfully loaded fine-tuned model")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def _load_lora_model(self, adapter_path: Path):
        """
        Load LoRA adapter model
        
        LoRA requires:
        1. Loading the base model (amazon/chronos-2) first
        2. Then loading the LoRA adapter
        """
        try:
            from peft import PeftModel
            import json
            
            # Read adapter_config.json to get the base model path
            with open(adapter_path / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "amazon/chronos-2")
            logger.info(f"Loading base model: {base_model_name}")
            
            # ✅ Load base model first
            base_pipeline = BaseChronosPipeline.from_pretrained(
                base_model_name,
                device_map=self.device,
            )
            
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            
            # ✅ Load LoRA adapter
            lora_model = PeftModel.from_pretrained(
                base_pipeline.model,
                str(adapter_path),
            )
            
            # ✅ Merge adapter into base model (optional, improves inference speed)
            # lora_model = lora_model.merge_and_unload()
            
            # Replace the model in the pipeline
            base_pipeline.model = lora_model
            self.pipeline = base_pipeline
            
            logger.info("✅ Successfully loaded LoRA adapter")
            
        except ImportError:
            raise ImportError(
                "LoRA adapter requires 'peft' library. Install with: pip install peft"
            )

    def _prepare_inputs(self, context: Union[np.ndarray, List[np.ndarray]]) -> List[Dict]:
        """
        Convert numpy array to Chronos-2 inputs format
        
        Args:
            context: Single sequence (1D) or list of multiple sequences
            
        Returns:
            List[Dict]: [{"target": np.ndarray}, ...]
        """
        # Convert to list format
        if isinstance(context, np.ndarray):
            if context.ndim == 1:
                # Single sequence
                context_list = [context]
            elif context.ndim == 2:
                # Multiple sequences
                context_list = [context[i] for i in range(context.shape[0])]
            else:
                raise ValueError(f"Unsupported context shape: {context.shape}")
        else:
            context_list = context

        # Convert to Chronos-2 format
        inputs = [{"target": np.asarray(ts, dtype=np.float32)} for ts in context_list]
        return inputs

    def predict_quantiles(
            self,
            context: Union[np.ndarray, List[np.ndarray]],
            prediction_length: int = 12,
            quantile_levels: List[float] = [0.1, 0.5, 0.9],
    ) -> np.ndarray:
        """
        Get prediction quantiles (recommended usage)
        
        Chronos-2's predict_quantiles returns a tuple:
        - result[0]: All quantiles, shape (num_series, prediction_length, num_quantiles) ✅ Use this
        - result[1]: Median only, shape (num_series, prediction_length)
        
        Args:
            context: Historical data
            prediction_length: Prediction length
            quantile_levels: List of quantile levels
            
        Returns:
            quantiles: shape (num_series, num_quantiles, prediction_length)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")

        # Prepare input format
        inputs = self._prepare_inputs(context)

        # Chronos-2 predict_quantiles method
        result = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        # ✅ result is a tuple: (all quantiles, median only)
        if isinstance(result, tuple):
            quantiles_list = result[0]
        else:
            quantiles_list = result

        # Convert to numpy array
        converted = []
        for q in quantiles_list:
            if torch.is_tensor(q):
                converted.append(q.cpu().numpy())
            elif isinstance(q, np.ndarray):
                converted.append(q)
            else:
                converted.append(np.asarray(q))
        
        # Concatenate all sequences: (num_series, prediction_length, num_quantiles)
        quantiles = np.concatenate(converted, axis=0)
        
        # Convert to target format: (num_series, num_quantiles, prediction_length)
        quantiles = quantiles.swapaxes(-1, -2)
        
        return quantiles

    def predict_median(
            self,
            context: Union[np.ndarray, List[np.ndarray]],
            prediction_length: int = 12,
    ) -> np.ndarray:
        """
        Get median prediction (0.5 quantile)
        
        Args:
            context: Historical data
            prediction_length: Prediction length
            
        Returns:
            predictions: shape (num_series, prediction_length)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")

        # Prepare input format
        inputs = self._prepare_inputs(context)

        # Chronos-2 predict_quantiles method
        result = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
        )

        # ✅ Use result[1] to directly get the median
        if isinstance(result, tuple):
            median_list = result[1]
        else:
            median_list = result[0] if isinstance(result, list) else [result]

        # Convert to numpy array
        converted = []
        for m in median_list:
            if torch.is_tensor(m):
                converted.append(m.cpu().numpy())
            elif isinstance(m, np.ndarray):
                converted.append(m)
            else:
                converted.append(np.asarray(m))
        
        # Concatenate: (num_series, prediction_length)
        median_predictions = np.concatenate(converted, axis=0)
        
        # If there is only one sequence, return 1D array
        if median_predictions.shape[0] == 1:
            return median_predictions[0]
        
        return median_predictions

    def predict(
            self,
            context: Union[np.ndarray, List[np.ndarray]],
            prediction_length: int = 12,
            num_samples: int = 20
    ) -> np.ndarray:
        """
        Get sample predictions (returns all samples)
        
        Args:
            context: Historical data
            prediction_length: Prediction length
            num_samples: Number of samples
            
        Returns:
            predictions: shape (num_series, num_samples, prediction_length)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")

        # Prepare input format
        inputs = self._prepare_inputs(context)

        # Chronos-2 predict method
        predictions = self.pipeline.predict(
            inputs=inputs,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        # Convert to numpy array
        if isinstance(predictions, (list, tuple)):
            converted = []
            for p in predictions:
                if torch.is_tensor(p):
                    converted.append(p.cpu().numpy())
                else:
                    converted.append(np.asarray(p))
            predictions = np.concatenate(converted, axis=0)

        return predictions

    def predict_df(
            self,
            df: pd.DataFrame,
            future_df: Optional[pd.DataFrame] = None,
            prediction_length: int = 12,
            quantile_levels: List[float] = [0.1, 0.5, 0.9],
            id_column: str = "item_id",
            timestamp_column: str = "timestamp",
            target: str = "target",
    ) -> pd.DataFrame:
        """
        Predict using DataFrame API (High-level API, recommended for predictions with covariates)
        
        Args:
            df: Long-format DataFrame containing id, timestamp, and target columns
            future_df: Optional future covariate DataFrame
            prediction_length: Prediction length
            quantile_levels: List of quantile levels
            id_column: ID column name
            timestamp_column: Timestamp column name
            target: Target column name
            
        Returns:
            Prediction result DataFrame
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")

        return self.pipeline.predict_df(
            df=df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
        )

    def save_model(self, save_path: str):
        """Save the model"""
        if self.pipeline is not None:
            logger.info(f"Saving model to {save_path}")
            self.pipeline.model.save_pretrained(save_path)
            logger.info(f"Model saved successfully")
        else:
            raise ValueError("No model to save")