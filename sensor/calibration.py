def weight_to_count(weight: float, unit_weight: float = 3.0, round_result: bool = False) -> float:
    """
    Converts a total weight to an estimated part count based on unit weight.
    """
    if unit_weight <= 0:
        raise ValueError("Unit weight must be strictly positive.")
    if weight is None or weight < 0:
        return 0.0
        
    count = weight / unit_weight
    if round_result:
        count = round(count)
    return float(count)
