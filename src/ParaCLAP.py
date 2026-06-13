import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model

from adapter import DEFAULT_EMOTIONS, audio_root, build_label_dataframe

AUDIO_ENCODER = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TEXT_ENCODER = "bert-base-uncased"
EMBEDDING_DIM = 1024
SAMPLE_RATE = 16000
CLIP_SECONDS = 5
NUM_EPOCHS = 100
BATCH_SIZE = 32
BASE_LR = 1e-3
ENCODER_LR_FACTOR = 100
CKPT_DIR = "ckpt"


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


class Projection(nn.Module):

    def __init__(self, d_in, d_out, p=0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.linear1(x)
        h = h + self.drop(self.linear2(F.gelu(h)))
        return self.norm(h)


class SpeechEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.base = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.base.config.hidden_size

    def forward(self, wav):
        return self.base(wav).last_hidden_state.mean(dim=1)


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.base.config.hidden_size

    def forward(self, tokens):
        with torch.no_grad():
            out = self.base(**tokens).last_hidden_state
        return out[:, 0, :].detach()


class CLAP(nn.Module):

    def __init__(self, speech_name, text_name, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.audio_branch = SpeechEncoder(speech_name)
        self.text_branch = TextEncoder(text_name)
        self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim)
        self.text_projection = Projection(self.text_branch.hidden_size, embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        for p in self.text_branch.parameters():
            p.requires_grad_(False)

    def forward(self, wav, tokens):
        audio = F.normalize(self.audio_projection(self.audio_branch(wav)), dim=-1)
        text = F.normalize(self.text_projection(self.text_branch(tokens)), dim=-1)
        return audio, text, self.logit_scale.exp()


def clip_loss(audio, text, scale):
    logits = scale * audio @ text.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class CaptionDataset(Dataset):

    def __init__(self, df, wav_root, caption_dir):
        self.df = df
        self.wav_root = wav_root
        self.caption_dir = caption_dir
        self.target_length = CLIP_SECONDS * SAMPLE_RATE

    def __len__(self):
        return len(self.df)

    def _load_audio(self, file):
        wav, sr = torchaudio.load(os.path.join(self.wav_root, file))
        if sr != SAMPLE_RATE:
            wav = F_audio.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
        wav = wav.mean(0, keepdim=True)
        n = wav.shape[1]
        if n > self.target_length:
            start = random.randint(0, n - self.target_length)
            wav = wav[:, start:start + self.target_length]
        else:
            wav = F.pad(wav, (0, self.target_length - n))
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        return wav.squeeze(0)

    def _load_caption(self, file):
        with open(os.path.join(self.caption_dir, f"{file}.json"), encoding="utf-8") as fp:
            return json.load(fp)["caption"]

    def __getitem__(self, index):
        file = self.df.index[index]
        return self._load_audio(file), self._load_caption(file)


def build_optimizer(model):
    audio_params = [p for n, p in model.named_parameters()
                    if n.startswith("audio_branch") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("audio_branch")
                   and not n.startswith("text_branch") and p.requires_grad]
    return torch.optim.Adam([
        {"params": audio_params, "lr": BASE_LR / ENCODER_LR_FACTOR},
        {"params": head_params, "lr": BASE_LR},
    ], lr=BASE_LR)


def train_one_epoch(model, loader, optimizer, tokenizer, device, epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    for i, (wav, captions) in enumerate(loader):
        wav = wav.to(device, non_blocking=True)
        tokens = tokenizer(list(captions), padding=True, truncation=True,
                           return_tensors="pt").to(device)

        audio, text, scale = model(wav, tokens)
        loss = clip_loss(audio, text, scale)

        if not torch.isfinite(loss):
            print(f"epoch {epoch} | batch {i} | non-finite loss, skipped", flush=True)
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        if i % 100 == 0:
            print(f"epoch {epoch} | batch {i}/{len(loader)} | loss {loss.item():.4f}", flush=True)
    return total_loss / max(steps, 1)


def save_ckpt(path, model, optimizer, epoch, best_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    setup_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"config: epochs={NUM_EPOCHS} batch={BATCH_SIZE} base_lr={BASE_LR} "
          f"enc_lr_factor={ENCODER_LR_FACTOR} emb={EMBEDDING_DIM} clip_s={CLIP_SECONDS}")

    dataset_root = "/nas/student/DavidFrank/MSP-Podcast"
    caption_dir = "captions_llm"

    df = build_label_dataframe(dataset_root, split="train", emotions=DEFAULT_EMOTIONS)
    n_total = len(df)
    has_caption = df.index.map(lambda f: os.path.exists(os.path.join(caption_dir, f"{f}.json")))
    df = df[has_caption]
    print(f"{len(df)}/{n_total} utterances with a cached caption ({n_total - len(df)} dropped)")

    loader = DataLoader(
        CaptionDataset(df, audio_root(dataset_root), caption_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM).to(device)
    optimizer = build_optimizer(model)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)

    os.makedirs(CKPT_DIR, exist_ok=True)
    last_path = os.path.join(CKPT_DIR, "last.pth.tar")
    best_path = os.path.join(CKPT_DIR, "best.pth.tar")

    start_epoch, best_loss = 1, float("inf")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        print(f"resuming at epoch {start_epoch} (best loss {best_loss:.4f})")

    run_start = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        avg_loss = train_one_epoch(model, loader, optimizer, tokenizer, device, epoch)
        model.logit_scale.data.clamp_(0, np.log(100))

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_ckpt(last_path, model, optimizer, epoch, best_loss)
        if is_best:
            save_ckpt(best_path, model, optimizer, epoch, best_loss)
            print(f"[{epoch}] new best (loss={best_loss:.4f})", flush=True)

        epoch_min = (time.time() - epoch_start) / 60
        sps = len(loader.dataset) / max(time.time() - epoch_start, 1e-9)
        done = epoch - start_epoch + 1
        eta_h = (time.time() - run_start) / done * (NUM_EPOCHS - epoch) / 3600
        scale = model.logit_scale.exp().item()
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"[{time.strftime('%H:%M:%S')}][{epoch}/{NUM_EPOCHS}] "
              f"avg loss {avg_loss:.4f} | scale {scale:.2f} | {sps:.1f} samp/s | "
              f"peak {mem:.2f}GB | {epoch_min:.1f} min | eta {eta_h:.1f}h", flush=True)


if __name__ == "__main__":
    main()
