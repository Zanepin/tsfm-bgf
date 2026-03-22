import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chronos import ChronosPipeline
import numpy as np
from typing import Optional, List, Union


class ChronosT5Model:
    def __init__(self, model_name: str = "amazon/chronos-t5-base", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load pre-trained model"""
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            )
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self,
                context: Union[np.ndarray, List[np.ndarray]],
                prediction_length: int = 12,
                num_samples: int = 20) -> np.ndarray:
        """Predict time series"""
        if self.pipeline is None:
            raise ValueError("Model not loaded")

        # Ensure correct input format
        if isinstance(context, np.ndarray):
            if context.ndim == 1:
                context = [context]
            elif context.ndim == 2:
                context = [context[i] for i in range(context.shape[0])]

        # Use Chronos for prediction
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )

        return forecast

    def forecast_mean(self,
                      context: Union[np.ndarray, List[np.ndarray]],
                      prediction_length: int = 12,
                      num_samples: int = 20) -> np.ndarray:
        """Get prediction mean"""
        forecast = self.predict(context, prediction_length, num_samples)

        # Calculate mean prediction
        if len(forecast) == 1:
            return forecast[0].mean(axis=0)
        else:
            return np.array([f.mean(axis=0) for f in forecast])

    def save_model(self, save_path: str):
        """Save model"""
        if self.pipeline is not None:
            self.pipeline.model.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    def load_finetuned_model(self, model_path: str):
        """Load fine-tuned model"""
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            )
            print(f"Successfully loaded fine-tuned model from {model_path}")
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            raise



