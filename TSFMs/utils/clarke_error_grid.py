"""
Shared utility module for TSFMs evaluation.
Provides the Clarke Error Grid Analysis (CEA) used across all evaluation scripts.
"""

import numpy as np


class ClarkeErrorGrid:
    """Clarke Error Grid Analysis for blood glucose prediction evaluation."""

    def __init__(self):
        self.zone_colors = {
            'A': 'lightgreen', 'B': 'lightblue',
            'C': 'orange', 'D': 'red', 'E': 'red'
        }

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
            elif (ref >= 240 and (pred >= 70 and pred <= 180)) or (
                    ref <= 175 / 3 and pred <= 180 and pred >= 70) or (
                    (ref >= 175 / 3 and ref <= 70) and pred >= (6 / 5) * ref):
                zones.append('D')
            else:
                zones.append('B')
        return np.array(zones)

    def run(self, ref_values, pred_values, plot=False):
        ref = np.array(ref_values).flatten()
        pred = np.array(pred_values).flatten()
        if len(ref) != len(pred):
            return {'error': 'Shape mismatch'}
        zones = self._calculate_zones(ref, pred)
        counts = {z: np.sum(zones == z) for z in ['A', 'B', 'C', 'D', 'E']}
        total = len(ref)
        if total == 0:
            return {'error': 'No data points'}
        zone_ab_count = counts['A'] + counts['B']
        zone_cde_count = counts['C'] + counts['D'] + counts['E']
        return {
            'total_points': total,
            'AB_percentage': (zone_ab_count / total) * 100,
            'CDE_percentage': (zone_cde_count / total) * 100
        }


def calculate_clarke_metrics(ref_values, pred_values):
    """Convenience function for Clarke Error Grid evaluation."""
    analyzer = ClarkeErrorGrid()
    return analyzer.run(ref_values, pred_values, plot=False)
