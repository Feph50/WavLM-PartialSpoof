import fnmatch
import os
import random
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class PartialSpoofDataset(Dataset):
    """
    PartialSpoof Dataset for Speech Editing Detection and Localization.
    Encapsulates audio file indexing, segment label loading, waveform reading,
    and resolution-aware padding/cropping.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
        resolution: float = 0.16,
        label_root: Optional[str] = None,
        max_label_len: int = 25,
        pad_mode: str = "train",
        debug_mode: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.label_root = label_root if label_root is not None else os.path.join(root, "segment_labels")
        self.max_label_len = max_label_len
        self.pad_mode = pad_mode
        self.debug_mode = debug_mode
        self.max_samples = max_samples
        self.scale = int(self.sample_rate * self.resolution)

        # Map standard split names to PartialSpoof directory/label conventions
        if self.split in ("val", "dev"):
            self.split_dir = "dev"
        elif self.split in ("test", "eval"):
            self.split_dir = "eval"
        else:
            self.split_dir = self.split

        # 1. Load label dictionary
        self.labels = self._load_labels()

        # 2. Find audio files
        split_audio_dir = os.path.join(self.root, self.split_dir)
        all_audio_files = self._find_audio_files(split_audio_dir, query="*.wav")
        assert len(all_audio_files) > 0, f"No wav files found in {split_audio_dir}"

        # 3. Filter valid samples
        self.sample_list = []
        for f in all_audio_files:
            utt_id = os.path.splitext(os.path.basename(f))[0]
            if utt_id in self.labels:
                self.sample_list.append(f)

        self.sample_list = sorted(self.sample_list)
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.sample_list]
        print(f"[{self.split}] Loaded {len(self.sample_list)} valid samples out of {len(all_audio_files)} files.")

        # 4. Debug subsampling
        if self.debug_mode and self.max_samples is not None:
            if len(self.sample_list) > self.max_samples:
                print(f"[{self.split}] Debug mode: limiting samples to {self.max_samples}")
                self.sample_list = self.sample_list[:self.max_samples]
                self.utt_ids = self.utt_ids[:self.max_samples]

    @staticmethod
    def _find_audio_files(root_dir: str, query: str = "*.wav") -> List[str]:
        """Finds all audio files matching query pattern recursively."""
        files = []
        for root, _, filenames in os.walk(root_dir, followlinks=True):
            for filename in fnmatch.filter(filenames, query):
                files.append(os.path.join(root, filename))
        return sorted(files)

    def _load_labels(self) -> Dict[str, np.ndarray]:
        """Loads segment labels dictionary from .npy file."""
        label_file = os.path.join(self.label_root, f"{self.split_dir}_seglab_{self.resolution}.npy")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")
        raw_labels = np.load(label_file, allow_pickle=True).item()
        return {k: v.astype(int) for k, v in raw_labels.items()}

    def _load_audio(self, path: str) -> np.ndarray:
        """Reads single-channel audio waveform."""
        audio, sr = sf.read(path)
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr} for {path}")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio

    def _pad_or_crop(
        self, utt_id: str, audio: torch.Tensor, label: torch.Tensor, label_len: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """Applies padding or random cropping based on pad_mode."""
        scale = self.scale

        if self.pad_mode == "train":
            target_audio_len = self.max_label_len * scale
            if label_len < self.max_label_len:
                if len(audio) < target_audio_len:
                    audio = F.pad(audio, (0, target_audio_len - len(audio)), mode="constant", value=0.0)
                else:
                    audio = audio[:target_audio_len]
                label = F.pad(label, (0, self.max_label_len - label_len), mode="constant", value=0.0)
                effective_len = label_len
            else:
                start = random.randint(0, label_len - self.max_label_len)
                start_sample = start * scale
                end_sample = (start + self.max_label_len) * scale
                audio = audio[start_sample:end_sample]
                if len(audio) < target_audio_len:
                    audio = F.pad(audio, (0, target_audio_len - len(audio)), mode="constant", value=0.0)
                label = label[start : start + self.max_label_len]
                effective_len = self.max_label_len
            return utt_id, audio, label, effective_len
        else:
            target_audio_len = label_len * scale
            if len(audio) < target_audio_len:
                audio = F.pad(audio, (0, target_audio_len - len(audio)), mode="constant", value=0.0)
            elif len(audio) > target_audio_len:
                audio = audio[:target_audio_len]
            return utt_id, audio, label, label_len

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        utt_id = self.utt_ids[index]
        audio = torch.from_numpy(self._load_audio(self.sample_list[index])).float()
        label = torch.from_numpy(self.labels[utt_id]).float()
        return self._pad_or_crop(utt_id, audio, label, len(label))

    def __len__(self) -> int:
        return len(self.utt_ids)

    @staticmethod
    def collate_fn(batch: List[Tuple[str, torch.Tensor, torch.Tensor, int]]) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        utt_ids, inputs, labels, label_lens = zip(*batch)

        # Dynamic padding for inputs [B, Max_Audio_Len]
        max_input_len = max(x.size(0) for x in inputs)
        padded_inputs = torch.stack([
            F.pad(x, (0, max_input_len - x.size(0)), value=0.0) if x.size(0) < max_input_len else x
            for x in inputs
        ], dim=0)

        # Dynamic padding for labels [B, Max_Label_Len]
        max_label_len = max(y.size(0) for y in labels)
        padded_labels = torch.stack([
            F.pad(y, (0, max_label_len - y.size(0)), value=0.0) if y.size(0) < max_label_len else y
            for y in labels
        ], dim=0)

        label_lens_tensor = torch.tensor(label_lens, dtype=torch.long)
        return list(utt_ids), padded_inputs, padded_labels, label_lens_tensor


class PartialSpoofDataModule(LightningDataModule):
    """
    Encapsulates DataLoaders for Train, Validation, and Test splits.
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        resolution: float = 0.16,
        max_label_len: int = 25,
        batch_size: int = 16,
        num_workers: int = 4,
        debug_mode: bool = False,
        max_samples: Optional[int] = None,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.max_label_len = max_label_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_mode = debug_mode
        self.max_samples = max_samples
        self.pin_memory = pin_memory

        self.train_dataset: Optional[PartialSpoofDataset] = None
        self.val_dataset: Optional[PartialSpoofDataset] = None
        self.test_dataset: Optional[PartialSpoofDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = PartialSpoofDataset(
                root=self.root,
                split="train",
                sample_rate=self.sample_rate,
                resolution=self.resolution,
                max_label_len=self.max_label_len,
                pad_mode="train",
                debug_mode=self.debug_mode,
                max_samples=self.max_samples,
            )
            self.val_dataset = PartialSpoofDataset(
                root=self.root,
                split="val",
                sample_rate=self.sample_rate,
                resolution=self.resolution,
                max_label_len=self.max_label_len,
                pad_mode="eval",
                debug_mode=self.debug_mode,
                max_samples=self.max_samples,
            )

        if stage in ("test", None):
            self.test_dataset = PartialSpoofDataset(
                root=self.root,
                split="test",
                sample_rate=self.sample_rate,
                resolution=self.resolution,
                max_label_len=self.max_label_len,
                pad_mode="eval",
                debug_mode=self.debug_mode,
                max_samples=self.max_samples,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=PartialSpoofDataset.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=PartialSpoofDataset.collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=PartialSpoofDataset.collate_fn,
            pin_memory=self.pin_memory,
        )
