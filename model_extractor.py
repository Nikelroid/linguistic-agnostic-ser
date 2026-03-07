import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerExtractor(nn.Module):
    """
    Extracts intermediate hidden states from pre-trained Speech Transformers 
    (e.g., wav2vec 2.0, HuBERT, WavLM).
    """
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        # Ensure we output all hidden states
        self.config.output_hidden_states = True
        
        # Load the model and freeze it
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.model.eval()
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_num_layers(self):
        """Returns the number of hidden layers (including initial embedding layer)."""
        # Usually num_hidden_layers + 1 (the initial CNN/embedding output)
        if hasattr(self.config, 'num_hidden_layers'):
            return self.config.num_hidden_layers + 1
        return len(self.model.config.hidden_sizes)

    def extract_from_waveform(self, waveform):
        """
        Forward pass to extract layer-wise representations.
        waveform: Tensor of shape (batch_size, sequence_length)
        Returns a list/tuple of hidden states across all layers as numpy arrays.
        """
        waveform = waveform.to(self.device)
        with torch.no_grad():
            outputs = self.model(waveform)
            # hidden_states is a tuple of (batch_size, sequence_length, hidden_size)
            hidden_states = outputs.hidden_states
        
        pooled_states = []
        for state in hidden_states:
            # Mean pooling over the time dimension (dim=1)
            pooled_state = torch.mean(state, dim=1)
            pooled_states.append(pooled_state.cpu().numpy())
            
        return pooled_states
