#dataset and dataloader for whisper model handles loading audio files and their transcriptions
import torch
from torch.utils.data import Dataset , DataLoader
from pathlib import Path
import pandas as pd
from typing import Tuple , List , Dict
import config
from data_preprocessing import AudioPreprocesser
class WhisperDataset(Dataset):
    '''
    Docstring for WhisperDataset
    custom dataset foir the whisper model loads audio files and their corressponding transcriptions
    '''
    def __init__(self , 
                 audio_paths: List[str],
                 transcripts: List[str],
                 preprocesser: AudioPreprocesser,
                 tokenizer,
                 is_training:bool = True):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.preprocesser = preprocesser
        self.tokenizer = tokenizer
        self.is_training = is_training
        #validate that we have matching audio and transcriptions
        assert len(audio_paths) == len(transcripts) , \
            "Number of audios must match number of transcripts"
    def __len__(self) -> int:
        '''
        Docstring for __len__
        
        :param self: Description
        :return: Description
        :rtype: int
        return the total number of samples in dataset
        '''
        return len(self.audio_paths)
    def __getitem__(self , idx:int) -> Dict[str , torch.Tensor]:
        '''
        Docstring for __getitem__
        
        :param self: Description
        :param idx: Description
        :type idx: int
        :return: Description
        :rtype: Dict[str, Tensor]
        '''
        #get audio path and transcript for index
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        #preprocess audio
        audio_features = self.preprocesser.process(
            audio_path,
            apply_augmentation = self.is_training
        )
        #tokenize transcript text
        #add special tokens: <start> + transcript + <end>
        tokens = self.tokenizer.encode(
            transcript,
            add_special_tokens=True,
            max_length = config.MAX_TEXT_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors="pt" #returns as pytorch 
        )
        #remove batch dimention 
        tokens = tokens.squeeze(0)
        #create sample dictionary
        sample = {
            'audio_features': audio_features ,
            'tokens':tokens,
            'transcript':transcript 
        }
        return sample 
def collate_fn(batch: List[Dict])-> Dict[str , torch.tensor]:
    '''
    Docstring for collate_fn
    
    :param batch: Description
    :type batch: List[Dict]
    :return: Description
    :rtype: Dict[str, Any]
    '''
    #extract audio features from all samples 
    audio_features = [item['audio_features'] for item in batch]
    #stack into batch 
    tokens = torch.stack(tokens , dim = 0 )
    #create attention masks 
    #for audio: mask out padding (all zeros are padding)
    audio_mask = (audio_features.sum(dim=-1) == 0).squeeze(1)
    #for tokens 
    token_mask = (tokens == config.PAD_TOKEN_ID)
    #extract orginal transcripts (debugging and logging)
    transcripts = [item["audio_features"] for item in batch]
    #create input tokens (all tokens except last)
    #this is what decoder recieves as input
    input_tokens = tokens[: , :-1]
    #create target tokens (all tokens except first )
    #this is what decoder should predicti and it will predict 
    target_tokens = tokens[: , 1:]
    input_tokens = input_tokens.transpose(0 , 1)
    target_tokens = target_tokens.transpose(0 , 1)
    
    #adjust token mask for the decoder input remove last token mask 
    decoder_mask = token_mask[:  ,  :-1]
    return {
        'audio_features': audio_features ,
        "input_tokens":input_tokens,
        "target_tokens":target_tokens,
        "audio_mask":audio_mask,
        "decoder_mask":decoder_mask,
        "transcripts":transcripts
    }    
def create_dataloaders(train_audio_paths:List[str] , 
                       train_transcripts:List[str],
                       val_audio_paths:List[str],
                       val_transcripts:List[str],
                       tokenizer,
                       batch_size:int = config.BATCH_SIZE,
                       num_workers:int = config.NUM_WORKERS) -> Tuple[DataLoader , DataLoader]:
    '''
    Docstring for create_dataloaders
    
    :param train_audio_paths: Description
    :type train_audio_paths: List[str]
    :param train_transcripts: Description
    :type train_transcripts: List[str]
    :param val_audio_paths: Description
    :type val_audio_paths: List[str]
    :param val_transcripts: Description
    :type val_transcripts: List[str]
    :param tokenizer: Description
    :param batch_size: Description
    :type batch_size: int
    :param num_workers: Description
    :type num_workers: int
    :return: Description
    :rtype: Tuple[DataLoader, DataLoader]
    '''
    #create audio processer
    preprocesser  = AudioPreprocesser()
    #create training dataset(with augmentation)
    train_dataset = WhisperDataset(
        audio_paths=train_audio_paths,
        transcripts=train_transcripts,
        preprocesser=preprocesser,
        tokenizer=tokenizer,
        is_training=True #enable augmentation
    )
    #create validation dataset (no augmentation)
    val_dataset = WhisperDataset(
        audio_paths=val_audio_paths,
        transcripts=val_transcripts,
        preprocesser=preprocesser,
        tokenizer=tokenizer,
        is_training=False #disable augmentation 
    )  
    #create training dataloader
    train_loAder = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, #shuffle training data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    #create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers = num_workers,
        collate_fn=collate_fn,#please dont shuffle validation data
        pin_memory=True,
        drop_last=False
        
    )
    print(f"Created Data Loaders:")
    print(f"training samples : {len(train_dataset)}")
    print(f"the total validation samples is {len(val_dataset)}")
    print(f"The total batch size is {batch_size}")
    print(f"training batches {len(train_loAder)}")
    print(f"Validation batches {len(val_loader)}")
    return train_loAder , val_loader


def load_data_from_csv(csv_path:str) -> Tuple[List[str] , List[str]]:
    '''
    Docstring for load_data_from_csv
    
    :param csv_path: Description
    :type csv_path: str
    :return: Description
    :rtype: Tuple[List[str], List[str]]
    '''
    import pandas as pd #importing the pandas library for loading of csv
    data = pd.read_csv(csv_path)
    #extracts audio paths and transcriptys 
    audio_paths = data['audio_path'].tolist()
    transcripts = data['transcripts'].tolist()
    #validate that files exist or not 
    valid_indices = []
    for idx , path in enumerate(audio_paths):
        if Path(path).exists():
            valid_indices.append(idx)
        else:
            print(f"caution the data wont exist")
    
    #keep only valid samples
    audio_paths =  [audio_paths[i] for i in valid_indices]
    transcripts = [transcripts[i] for i in valid_indices]
    return audio_paths , transcripts

#examples usasge
if __name__ == "__main__":
    from transformers import WhisperTokenizer
    #instalizing tokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    # Example: Create dummy data
    # In practice, you would load this from CSV files
    dummy_audio_paths = [str(config.RAW_DATA_DIR / f"audio{i}.wav") for i in range(10)]
    dummy_transcripts = [f"This is sample transcript number {i}" for i in range(10)]
    
    # Split into train and validation
    split_idx = int(len(dummy_audio_paths) * config.TRAIN_VAL_SPLIT)
    train_paths = dummy_audio_paths[:split_idx]
    train_transcripts = dummy_transcripts[:split_idx]
    val_paths = dummy_audio_paths[split_idx:]
    val_transcripts = dummy_transcripts[split_idx:]
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    
    
    print("\nTo test with real data:")
    print("1. Place audio files in:", config.RAW_DATA_DIR)
    print("2. Create a CSV with columns: audio_path, transcript")
    print("3. Use load_data_from_csv() to load your data")
    
    print("\n" + "=" * 80)
    
