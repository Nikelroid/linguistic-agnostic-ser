import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor

class TransformerExtractor(nn.Module):
    """
    Extracts intermediate hidden states from pre-trained Speech Transformers 
    (e.g., wav2vec 2.0, HuBERT, WavLM, Whisper).
    """
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name.lower()
        self.is_whisper = "whisper" in self.model_name
        
        print(f"Loading {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        # Ensure we output all hidden states
        self.config.output_hidden_states = True
        
        # Load the feature extractor for correct preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Load the model and freeze it
        try:
            # First attempt: Try standard load (it will use cache if available or download if needed)
            # We prefer safetensors to bypass the security vulnerability check in newer Transformers
            self.model = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        except Exception as e:
            print(f"Standard loading failed for {model_name} (likely network timeout). Switching to local cache mode...")
            try:
                # Second attempt: Force local files only to bypass the Hub check/vulnerability check entirely
                self.model = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=True)
            except Exception as e2:
                print(f"Critical error: Model {model_name} not found in local cache and Hub is unreachable.")
                raise e2
        self.model.eval()
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_num_layers(self):
        """Returns the number of hidden layers (including initial embedding layer)."""
        if self.is_whisper:
            return self.config.encoder_layers + 1
        if hasattr(self.config, 'num_hidden_layers'):
            return self.config.num_hidden_layers + 1
        return len(self.model.config.hidden_sizes)

    def extract_from_waveform(self, waveform, sample_rate=16000):
        """
        Forward pass to extract layer-wise representations.
        waveform: Tensor of shape (batch_size, sequence_length)
        Returns a list/tuple of hidden states across all layers as numpy arrays.
        """
        # Convert waveform to numpy list for feature extractor
        waveform_np = waveform.cpu().numpy()
        waveform_list = [w for w in waveform_np]
        
        # Prepare inputs using the correct feature extractor
        inputs = self.feature_extractor(
            waveform_list, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        
        # WhisperFeatureExtractor creates 'input_features' (log-mel spectograms), 
        # Wav2Vec2FeatureExtractor creates 'input_values' (raw waveforms)
        if self.is_whisper and "input_features" in inputs:
            model_inputs = {"input_features": inputs["input_features"].to(self.device)}
        elif "input_values" in inputs:
            model_inputs = {"input_values": inputs["input_values"].to(self.device)}
        else:
            # Fallback to unpacking everything
            model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.is_whisper:
                # Whisper is an encoder-decoder model; we probe the encoder's hidden states
                outputs = self.model.encoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)
            
            hidden_states = outputs.hidden_states
        
        pooled_states = []
        for state in hidden_states:
            # Mean pooling over the time dimension (dim=1)
            pooled_state = torch.mean(state, dim=1)
            pooled_states.append(pooled_state.cpu().detach().numpy().copy())
            
        del outputs
        del hidden_states
        del model_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return pooled_states
