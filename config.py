#we will configure some settings here 
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
LOGS_DIR = PROJECT_ROOT / "logs"
for dir_path in [RAW_DATA_DIR , PROCESSED_DATA_DIR , TRANSCRIPTS_DIR , 
                 CHECKPOINTS_DIR , BEST_MODEL_DIR , LOGS_DIR]:
    dir_path.mkdir(parents=True , exist_ok=True)
    
#setuping audio processing parameters    
# Sampling rate - number of audio samples per second (16kHz is standard for speech)
SAMPLE_RATE = 16000
# Number of mel filterbanks - captures frequency information
N_MELS = 80
# FFT window size - number of samples in each analysis window
N_FFT = 400
# Hop length - number of samples between successive frames
HOP_LENGTH = 160
# Maximum audio duration in seconds (we'll pad or truncate to this length)
MAX_AUDIO_LENGTH = 30
# Maximum number of frames after feature extraction
MAX_FRAMES = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)
#model architecture parameters
ENCODER_LAYERS = 6  # Number of transformer encoder layers
ENCODER_HEADS = 8  # Number of attention heads in encoder
ENCODER_DIM = 512  # Dimension of encoder hidden states
ENCODER_FFN_DIM = 2048  # Dimension of feed-forward network
# Decoder parameters (generates text transcription)
DECODER_LAYERS = 6  # Number of transformer decoder layers
DECODER_HEADS = 8  # Number of attention heads in decoder
DECODER_DIM = 512  # Dimension of decoder hidden states
DECODER_FFN_DIM = 2048  # Dimension of feed-forward network
# Dropout rate - randomly drops connections to prevent overfitting
DROPOUT = 0.1
# Vocabulary size - number of unique tokens (characters/words)
VOCAB_SIZE = 51865  # Default for multilingual tokenizer
# Maximum sequence length for text output
MAX_TEXT_LENGTH = 448
# Batch size - number of samples processed together
BATCH_SIZE = 8
# Number of epochs - complete passes through the training data
NUM_EPOCHS = 50
# Learning rate - step size for gradient descent optimization
LEARNING_RATE = 1e-4
# Weight decay - L2 regularization to prevent overfitting
WEIGHT_DECAY = 0.01
# Gradient clipping - prevents exploding gradients
MAX_GRAD_NORM = 1.0
# Warmup steps - gradually increase learning rate at the start
WARMUP_STEPS = 1000
# How often to save checkpoints (in steps)
SAVE_CHECKPOINT_EVERY = 1000
# How often to evaluate on validation set (in steps)
EVAL_EVERY = 500
# Early stopping patience - stop if no improvement for N evaluations
EARLY_STOPPING_PATIENCE = 5
# Device - use GPU if available, otherwise CPU
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

USE_AUGMENTATION = True

# Time stretching - speed up or slow down audio
TIME_STRETCH_RATE = (0.8, 1.2)  # Range for random time stretching

# Pitch shifting - change audio pitch
PITCH_SHIFT_STEPS = (-2, 2)  # Range in semitones

# Background noise - add noise to audio
NOISE_FACTOR = 0.005  # Amount of random noise to add
# Special tokens used in text processing
PAD_TOKEN = "<|pad|>"  # Padding token for sequences
START_TOKEN = "<|startoftranscript|>"  # Marks beginning of transcript
END_TOKEN = "<|endoftext|>"  # Marks end of transcript
UNK_TOKEN = "<|unk|>"  # Unknown token for out-of-vocabulary words

# Token IDs (will be assigned by tokenizer)
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================
# Beam search width - number of candidates to keep during decoding
BEAM_SIZE = 5

# Length penalty - encourages longer or shorter outputs
LENGTH_PENALTY = 1.0

# Temperature - controls randomness in generation (lower = more deterministic)
TEMPERATURE = 0.0


# Train/validation split ratio
TRAIN_VAL_SPLIT = 0.9  # 90% training, 10% validation

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of workers for data loading
NUM_WORKERS = 4
def print_config():
    print("All parameter are configured properly")
    print(f"the project root directory is {PROJECT_ROOT}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Encoder Layers: {ENCODER_LAYERS}")
    print(f"Decoder Layers: {DECODER_LAYERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {NUM_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("whoo!! sangam all config is set you are good to go !")
    
if __name__ == "__main__":
    print_config()    