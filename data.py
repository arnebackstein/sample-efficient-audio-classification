import torch
import torchaudio
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class ESC50Dataset:
    def __init__(self, ds):
        self.ds = ds
        self.ds.set_format("torch")
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

    def process_audio(self, audio_dict):
        waveform = audio_dict["array"].float()
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec_db = (mel_spec_db + 100) / 100
        return mel_spec_db

    def get_data(self):
        mel_spectrograms = []
        for audio in self.ds["audio"]:
            mel_spec_db = self.process_audio(audio)
            mel_spectrograms.append(mel_spec_db)
        return torch.stack(mel_spectrograms), self.ds["target"]

def get_train_test_split():
    ds = load_dataset("ashraq/esc50", split="train")
    ds.set_format("torch")
    
    train_indices, test_indices = train_test_split(
        range(len(ds)),
        test_size=0.2,
        stratify=ds["target"],
        random_state=42
    )
    
    train_ds = ds.select(train_indices)
    test_ds = ds.select(test_indices)
    
    return ESC50Dataset(train_ds), ESC50Dataset(test_ds) 