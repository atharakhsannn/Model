from typing import Tuple, Optional

def decide(weight_count: Optional[float], model_count: Optional[float], tolerance: float) -> Tuple[str, Optional[float]]:
    """
    Makes a fusion decision based on the difference between weight count and model count.
    Returns status ('OK', 'NG', or 'UNKNOWN') and the absolute difference.
    """
    if weight_count is None or model_count is None:
        return "UNKNOWN", None
        
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")
        
    difference = abs(weight_count - model_count)
    if difference <= tolerance:
        status = "OK"
    else:
        status = "NG"
        
    return status, difference
