import abc
import numpy as np
from typing import Dict, Any

class BaseModelWrapper(abc.ABC):
    """
    Base wrapper interface for all time series benchmark models (Tsline).
    Forces all subclasses to implement train and evaluate methods uniformly 
    and generate output using a consistent metrics format.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def train(self, dataset_type: str, pred_len: int) -> Any:
        """
        Train the model
        :param dataset_type: Dataset name, e.g., 'op', 're', 'lg'
        :param pred_len: Prediction length
        :return: Trained model instance or model checkpoint path
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset_type: str, pred_len: int) -> Dict[str, Any]:
        """
        Evaluate the model
        Uses sliding window prediction to calculate metrics such as MAE, RMSE, and Clarke Error Grid.
        :param dataset_type: Dataset name, e.g., 'op', 're', 'lg'
        :param pred_len: Prediction length
        :return: A dictionary containing summarized standard evaluation metrics
        """
        pass
        
    def format_metrics_summary(self, dataset_type: str, pred_len: int, metrics_storage: Dict[str, list], subject_runtimes: list) -> Dict[str, Any]:
        """
        Uniform metrics formatting method provided for all subclasses.
        Takes in detailed subject-level metric lists and runtimes, and outputs 
        the final summary dictionary format to be written to a CSV.
        """
        def fmt(key):
            val_list = metrics_storage.get(key, [])
            if not val_list: 
                return "N/A"
            return f"{np.mean(val_list):.2f} ± {np.std(val_list):.2f}"
            
        median_runtime = np.median(subject_runtimes) if subject_runtimes else 0.0
        horizon_min = pred_len * 5 # Default 5-minute sampling frequency
        
        return {
            "Dataset": dataset_type.upper(),
            "Horizon (min)": horizon_min,
            "MAE": fmt("MAE"),
            "RMSE": fmt("RMSE"),
            "R2 (%)": fmt("R2"),
            "CEA Zone A+B (%)": fmt("CEA_AB"),
            "CEA Zone C+D+E (%)": fmt("CEA_CDE"),
            "Median Runtime (s)": f"{median_runtime:.4f}"
        }