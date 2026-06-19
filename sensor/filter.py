from typing import List, Optional

def smooth_weight(values: List[float], window_size: int = 5, use_ema: bool = False) -> Optional[float]:
    """
    Applies an average filter to a list of weight values.
    Supports simple moving average and exponential moving average (EMA).
    Returns None if the list is empty.
    """
    if not values:
        return None
        
    # Protection for small buffers
    actual_window = min(len(values), window_size)
    recent_values = values[-actual_window:]
    
    if use_ema:
        alpha = 2.0 / (actual_window + 1.0)
        ema = recent_values[0]
        for value in recent_values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema
    else:
        return sum(recent_values) / actual_window
