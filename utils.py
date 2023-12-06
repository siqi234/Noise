import torch

def save_model(model, path):
    """
    Save a PyTorch model to a file.

    Args:
    - model (torch.nn.Module): The model to save.
    - path (str): Path to the file where the model should be saved.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """
    Load a PyTorch model from a file.

    Args:
    - model (torch.nn.Module): The model instance to load the weights into.
    - path (str): Path to the file where the model is saved.
    - device (torch.device): The device to load the model onto.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

# Additional utility functions can be added here as needed
