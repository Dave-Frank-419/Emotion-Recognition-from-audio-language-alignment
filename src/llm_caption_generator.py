import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter import DEFAULT_EMOTIONS, build_label_dataframe

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 96
BATCH_SIZE = 8
LOG_EVERY_BATCHES = 10

SYSTEM_PROMPT = """
Role:
You are an expert audio-linguistic analyst and an emotive caption generator.

Task:
    Translate a list of short factual phrases describing one speech utterance
    into a highly vivid, natural, and cohesive descriptive caption (1-3 sentences).

Feature Logic:
    - Pitch (high / normal / low): high is bright/sharp, low is deep/heavy.
    - Pitch variation (high / normal / low): high is dynamic/expressive, low is flat/monotone.
    - Intensity (loud / normal / almost silent): loud is forceful/projecting, silent is muted/withdrawn.
    - Duration (long / medium / short): long is drawn-out/deliberate, short is clipped/abrupt.
    - Jitter (high / normal / low): high is trembling/unstable (stress, crying); low is steady.
    - Shimmer (high / normal / low): high is rough/breathy/unstable; low is clean/resonant.

Anchor Rule:
    - The caption MUST BEGIN with this exact sentence, verbatim:
      "The speaker is feeling <EMOTION>." (insert the provided emotion label)
    - Continue with 1-2 vivid sentences that must NOT repeat the emotion word.

Constraints:
    1. NEVER mention numbers, statistics, or "quantile".
    2. DO NOT list the input phrases mechanically; rewrite them as flowing prose.
    3. FUSE features into a cinematic description capturing physical sound and psychological state.
    4. Seamlessly weave the provided Gender label into the continuation.
    5. Strictly under 40 words in total.

Output Rule
    - Output ONLY the final caption. No conversational filler.
    - No explanations, no JSON, no "Here is the caption."

# Example
INPUT: {"file_id": "VOICE_001.wav",
        "emotion": "sadness",
        "gender": "female",
        "template_captions":
        ["has a low pitch", "has a low pitch variation", "is almost silent", "is of average length",
        "has a high jitter", "has a high shimmer", "emotion is sadness", "a female is speaking"]}

OUTPUT: The speaker is feeling sadness. The woman's voice is a heavy, somber murmur, its slow, hesitant
        pace and fragile, breathy instability betraying the weight she carries.
"""


def clean_caption(text):
    text = text.strip()
    text = re.sub(r"^(OUTPUT|Caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    if re.match(r"^(we are given|we must|let me|first,|step \d)", text,
                re.IGNORECASE):
        return ""
    return text.strip().strip('"').strip()


def build_user_prompt(file_id, emotion, gender, template_captions):
    payload = {
        "file_id": file_id,
        "emotion": emotion,
        "gender": gender,
        "template_captions": list(template_captions or []),
    }
    return "Input:\n" + json.dumps(payload, ensure_ascii=False)


class LLMCaptioner:

    def __init__(self, model_name=MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        print(f"captioner: {model_name} dtype={'bf16' if use_bf16 else 'fp32'} "
              f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
              flush=True)

    def _prompt(self, file_id, emotion, gender, templates):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": build_user_prompt(file_id, emotion, gender, templates)},
        ]
        return self.tok.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    def caption_batch(self, items):
        prompts = [self._prompt(*it) for it in items]
        enc = self.tok(prompts, return_tensors="pt",
                       padding=True).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [clean_caption(t) for t in decoded]


def generate_captions(dataset_root, template_dir, cache_dir, split="train",
                      emotions=None, batch_size=BATCH_SIZE, limit=None):
    df = build_label_dataframe(dataset_root, split=split, emotions=emotions)
    if limit is not None:
        df = df.iloc[:limit]
    os.makedirs(cache_dir, exist_ok=True)

    todo = []
    cached = 0
    missing_templates = 0
    for file, row in df.iterrows():
        if os.path.exists(os.path.join(cache_dir, f"{file}.json")):
            cached += 1
            continue
        tpl_path = os.path.join(template_dir, f"{file}.json")
        if not os.path.exists(tpl_path):
            missing_templates += 1
            continue
        with open(tpl_path, encoding="utf-8") as fp:
            templates = json.load(fp)
        todo.append((file, row["emotion"], row["gender"], templates))

    print(f"config: model={MODEL_NAME} max_new_tokens={MAX_NEW_TOKENS} batch={batch_size}", flush=True)
    print(f"split={split}: {len(df)} utterances, {cached} cached, "
          f"{missing_templates} without template json, {len(todo)} to caption", flush=True)
    if not todo:
        return

    captioner = LLMCaptioner()
    written = 0
    empty = 0
    start = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        captions = captioner.caption_batch(batch)
        for (file, emotion, *_), caption in zip(batch, captions):
            if not caption.startswith(f"The speaker is feeling {emotion}."):
                empty += 1
                continue
            with open(os.path.join(cache_dir, f"{file}.json"), "w",
                      encoding="utf-8") as fp:
                json.dump({"caption": caption}, fp, ensure_ascii=False, indent=2)
            written += 1
        done = i + len(batch)
        if (i // batch_size) % LOG_EVERY_BATCHES == 0:
            rate = done / max(time.time() - start, 1e-9)
            eta_h = (len(todo) - done) / max(rate, 1e-9) / 3600
            print(f"{done}/{len(todo)} | {rate:.2f} captions/s | eta {eta_h:.1f}h | "
                  f"empty {empty} | {captions[0]!r}", flush=True)
    print(f"finished: {written} captions written, {empty} empty outputs skipped", flush=True)


if __name__ == "__main__":
    dataset_root = "/nas/student/DavidFrank/MSP-Podcast"
    generate_captions(
        dataset_root,
        template_dir="templates",
        cache_dir="captions_llm_anchor",
        split="train",
        emotions=DEFAULT_EMOTIONS,
        limit=None,
    )
