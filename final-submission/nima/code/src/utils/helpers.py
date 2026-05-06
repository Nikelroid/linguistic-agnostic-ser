import numpy as np

def calculate_information_increase(rmse_scores):
    """
    Calculates the 'information increase' metric across layers.
    Lower RMSE means better feature prediction (more information).
    This function computes the relative drop in RMSE from the first layer.
    """
    if len(rmse_scores) == 0:
        return []
        
    base_rmse = rmse_scores[0]
    increases = []
    for rmse in rmse_scores:
        if base_rmse == 0:
            increases.append(0.0)
        else:
            inc = ((base_rmse - rmse) / base_rmse) * 100
            increases.append(inc)
    return increases
