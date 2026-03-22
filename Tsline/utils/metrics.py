import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ClarkeErrorGrid:
    def __init__(self):
        pass

    def _calculate_zones(self, ref_values, pred_values):
        zones = []
        for i in range(len(ref_values)):
            ref = ref_values[i]
            pred = pred_values[i]
            if (ref <= 70 and pred <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
                zones.append('A')
            elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
                zones.append('E')
            elif ((ref >= 70 and ref <= 290) and pred >= ref + 110) or (
                    (ref >= 130 and ref <= 180) and (pred <= (7 / 5) * ref - 182)):
                zones.append('C')
            elif (ref >= 240 and (pred >= 70 and pred <= 180)) or (ref <= 175 / 3 and pred <= 180 and pred >= 70) or (
                    (ref >= 175 / 3 and ref <= 70) and pred >= (6 / 5) * ref):
                zones.append('D')
            else:
                zones.append('B')
        return np.array(zones)

    def run(self, ref_values, pred_values):
        ref = np.array(ref_values).flatten()
        pred = np.array(pred_values).flatten()
        if len(ref) != len(pred) or len(ref) == 0: 
            return {'AB_percentage': 0, 'CDE_percentage': 0}
            
        zones = self._calculate_zones(ref, pred)
        counts = {z: np.sum(zones == z) for z in ['A', 'B', 'C', 'D', 'E']}
        total = len(ref)
        
        return {
            'AB_percentage': ((counts['A'] + counts['B']) / total) * 100,
            'CDE_percentage': ((counts['C'] + counts['D'] + counts['E']) / total) * 100
        }

def calculate_clarke_metrics(ref, pred):
    return ClarkeErrorGrid().run(ref, pred)

def calculate_all_metrics(y_true, y_pred) -> dict:
    """
    Unified calculation of all regression metrics and Clarke Error Grid (CEA)
    Input: One-dimensional array or list of true values and predicted values
    Output: Dictionary structure containing core medical and statistical evaluation indicators
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Fill in existing NaNs to avoid calculation crashes. Use the mean of true values to ensure the baseline does not adversely affect the test
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_true))
        
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) * 100
    clarke = calculate_clarke_metrics(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CEA_AB": clarke["AB_percentage"],
        "CEA_CDE": clarke["CDE_percentage"]
    }
