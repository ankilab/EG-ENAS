
def compute_model_size(model):
    #total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # Convert size to megabytes (optional)
    total_size_mb = total_size / (1024 ** 2)
    return  total_size_mb