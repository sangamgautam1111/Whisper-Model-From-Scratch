import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#now building whisper model architecture

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math
import config

class PositionEncoding(nn.Module):
    #adds positonal informations to input embeddings
    #since transformers dont have inhgerent notation of sequence order
    #we add positional encodings to give the model information about token
    #positions
     
    def __init__(self , d_model:int , max_len:int = 5000):
        #D:model = dimention of model embeddings
        #max_length: max sequence length
        
        super().__init__()
        #create a matrix where we can hold positional encoding(max_len , d_model)
        position = torch.arange(max_len).unsqueeze(1)#shape (max_length , 1)
        #this division term for sinusoidal encoding
        #thisd creates different frequency for different dimentition
        div_term = torch.exp(torch.arange(0 , d_model , 2) * (-math.log(10000.0) / d_model)) 
        #instalize positional encoding matrix
        pe  = torch.zeros(max_len , 1 , d_model)
        #apply sin to even indices 
        pe[: , 0 , 0::2] = torch.sin(position * div_term)
        #apply cos to odd incides
        pe[: , 0 , 1 :: 2]  = torch.cos(position * div_term)
        #rgiser as buffer
        self.register_buffer('pe' , pe)
        
    def forward(self , x:torch.Tensor) -> torch.Tensor:
        #add positional encoding to input tensor 
        #agrs = x: input tensor of shape(seq_len , batch  , d_model)
        #returns x with postional encoding added 
        x = x + self.pe[:x.size(0)]
        return x

class AudioEncoder(nn.Module):
    '''
    encoder that process audio featureas (mel - spectogram)
    converts audio features into  contextualized representation
    '''
    #calling init function
    def __init__(self):
        super().__init__()
        #add intial convolution layers to process mel - spectrogram
        #this reduces the time dimention and extracts low-level features
        self.conv1 = nn.Conv1d(
            in_channels = config.N_MELS,#takes input as mel freq
            out_channels = config.ENCODER_DIM,#gives output as model dimentition
            kernel_size = 3, #window size for convolution
            padding=1 #padding to maintatin sequence length
        )
        #now creating conv2 usually used for images but we will use to encode audio
        self.conv2 = nn.Conv1d(
            in_channels = config.ENCODER_DIM,
            out_channels = config.ENCODER_DIM,
            kernel_size=3,
            stride=2,
            padding=1
        )
        #postitional encoding to add sequence position information
        self.positional_encoding = PositionEncoding(
            d_model = config.ENCODER_DIM,
            max_len=config.MAX_FRAMES
        )
        #transformers encoder layers
        #this learns contextual relationships between different time stamps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.ENCODER_DIM, #model dimention
            nhead = config.ENCODER_HEADS,
            dim_feedforward = config.ENCODER_FFN_DIM,
            dropout = config.DROPOUT,
            batch_first = False
        )  
        #stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = config.ENCODER_LAYERS
        )
        #layer normalization for stable 
        self.layer_norm = nn.LayerNorm(config.ENCODER_DIM)
        
    def forward(self , x:torch.Tensor , mask: torch.Tensor = None) -> torch.Tensor:
        '''
        Docstring for forward
        
        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        :param mask: Description
        :type mask: torch.Tensor
        :return: Description
        :rtype: Tensor
        
        '''
        #remove channel dimention and transpose for conv1d
        x = x.squeeze(1).transpose(1 , 2 )
        #apply first convolution  with gelu activation 
        #if you dont understand what is gelu dont worry sangam is here
        # GELU is a smooth activation function that works well for transformers
        x = F.gelu(self.conv1(x))
        #apply second convolution 
        x = F.gelu(self.conv2(x))
        #transpose to time batch features   for transformer
        x = x.transpose(1 , 2).transpose(0 , 1)
        #add positional encoding
        x = self.positional_encoding(x)
        #create attention mask if padding mask  is provided
        #this prevents the model from attending to positions
        if mask is not None:
            #convert to bool mask to float mask for transformer
            # True values (padding) become -inf, False values become 0
            mask = mask.float().masked_fill(mask==0  , float('-inf')).masked_fill(mask==1 , float(0.0))
        #just pass through transformer encoder layers
        encoded = self.transformer_encoder(x , src_key_padding_mask=mask)
        #now applying layer normalization 
        encoded = self.layer_norm(encoded)
        return encoded #returning the encoded layer

#now creating class of text decoder which works to decode text
class TextDecoder(nn.Module):
    '''
    Docstring for TextDecoder
    decoder that generates text transcriptions from audio encoding
    uses cross-attention to attend to encoder outputs while generating texts
    
    '''
    
    #defining the init function
    def __init__(self , vocab_size : int = config.VOCAB_SIZE):
        '''
        Docstring for __init__
        
        :param self: Description
        :param vocab_size: Description
        :type vocab_size: int
        '''
        super().__init__()        
        self.vocab_size = vocab_size
        #token embediing layers
        #converts token ID to vectors
        self.embedding  = nn.Embedding(
            num_embeddings=vocab_size,#size of vocal 
            embedding_dim=config.DECODER_DIM#embedding dimention of decoder
        )
        #now doing posiitonal encoding for encoders
        self.positional_encoding = PositionEncoding(                             
            d_model = config.DECODER_DIM,
            max_len = config.MAX_TEXT_LENGTH
        )
        #transformers decoder layers
        #these generated text while attending to both previous tokens and audio 
        decoder_layer = nn.TransformerDecoderLayer(
          d_model  = config.DECODER_DIM,
          nhead = config.DECODER_HEADS,
          dim_feedforward=config.DECODER_FFN_DIM,
          dropout=config.DROPOUT,
          batch_first=False
        )
        #stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.DECODER_LAYERS
        )
        #output projection layer - converts decoder outputs to vocabulary logitis
        self.output_projection = nn.Linear(config.DECODER_DIM , vocab_size )
        #layer normalization  
        self.layer_norm  = nn.LayerNorm(config.DECODER_DIM)
        
    def generate_square_subsequent_mask(self , sz:int) -> torch.Tensor:
        """
        Generate casual mask to prevent attending to future tokens 
        this ensures the model can look at previous tokens only
        
        Args:
            sz: Sequence length
            
        Returns:
            mask: Upper triangular mask (sz, sz)
        """
        # Create mask where upper triangle is -inf (cannot attend)
        # and lower triangle + diagonal is 0 (can attend)
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the decoder
        
        Args:
            tgt: Target token IDs (seq_len, batch)
            memory: Encoder outputs (enc_seq_len, batch, encoder_dim)
            tgt_mask: Causal mask for target sequence
            tgt_key_padding_mask: Padding mask for target (batch, seq_len)
            memory_key_padding_mask: Padding mask for encoder (batch, enc_seq_len)
            
        Returns:
            logits: Output logits over vocabulary (seq_len, batch, vocab_size)
        """
        # Embed target tokens
        tgt_embedded = self.embedding(tgt) * math.sqrt(config.DECODER_DIM)
        
        # Add positional encoding
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            device = tgt.device
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device)
        
        # Pass through transformer decoder
        # The decoder attends to both previous tokens (self-attention)
        # and encoder outputs (cross-attention)
        decoded = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Apply layer normalization
        decoded = self.layer_norm(decoded)
        
        # Project to vocabulary size to get logits
        logits = self.output_projection(decoded)
        
        return logits


class WhisperModel(nn.Module):
    """
    Complete Whisper model combining encoder and decoder
    This is the main model class that handles speech-to-text conversion
    """
    
    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        super().__init__()
        
        # Initialize encoder (processes audio)
        self.encoder = AudioEncoder()
        
        # Initialize decoder (generates text)
        self.decoder = TextDecoder(vocab_size=vocab_size)
        
        # Store vocabulary size
        self.vocab_size = vocab_size
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights for better training stability
        Uses Xavier/Glorot initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                # For weight matrices, use Xavier uniform initialization
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                audio_features: torch.Tensor,
                target_tokens: torch.Tensor,
                audio_mask: torch.Tensor = None,
                target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the complete model
        
        Args:
            audio_features: Mel-spectrogram features (batch, 1, time, n_mels)
            target_tokens: Target token IDs (seq_len, batch)
            audio_mask: Padding mask for audio (batch, time)
            target_mask: Padding mask for target (batch, seq_len)
            
        Returns:
            logits: Predicted logits over vocabulary (seq_len, batch, vocab_size)
        """
        # Encode audio features
        memory = self.encoder(audio_features, mask=audio_mask)
        
        # Decode to text
        logits = self.decoder(
            tgt=target_tokens,
            memory=memory,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=audio_mask
        )
        
        return logits
    
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features only (for inference)
        
        Args:
            audio_features: Mel-spectrogram features
            
        Returns:
            memory: Encoded audio representations
        """
        return self.encoder(audio_features)
    
    def decode_step(self, 
                     target_tokens: torch.Tensor,
                     memory: torch.Tensor) -> torch.Tensor:
        """
        Single decoding step (for inference/generation)
        
        Args:
            target_tokens: Previously generated tokens
            memory: Encoded audio features
            
        Returns:
            logits: Logits for next token prediction
        """
        return self.decoder(target_tokens, memory)
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters
        
        Returns:
            total_params: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing and example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Whisper Model Architecture")
    print("=" * 80)
    
    # Create model instance
    model = WhisperModel(vocab_size=config.VOCAB_SIZE)
    
    # Print model information
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    # Create dummy data for testing
    batch_size = 2
    time_steps = 100
    seq_len = 50
    
    # Dummy audio features (batch, 1, time, n_mels)
    dummy_audio = torch.randn(batch_size, 1, time_steps, config.N_MELS)
    
    # Dummy target tokens (seq_len, batch)
    dummy_targets = torch.randint(0, config.VOCAB_SIZE, (seq_len, batch_size))
    
    print(f"\nInput audio shape: {dummy_audio.shape}")
    print(f"Target tokens shape: {dummy_targets.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():  # No gradients needed for testing
        output = model(dummy_audio, dummy_targets)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({seq_len}, {batch_size}, {config.VOCAB_SIZE})")
    
    print("\n" + "=" * 80)
    print("Model architecture test completed successfully!")
    print("=" * 80)             