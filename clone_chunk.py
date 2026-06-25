#!/usr/bin/env python3
"""
Voice-clone dubber for ONE chunk (no Sarvam).
Pipeline:
  1. Extract reference audio (the speaker's original voice) from the chunk.
  2. Transcribe source language with faster-whisper.
  3. Translate to target language (offline argos-translate).
  4. Generate target-language speech CLONED from the reference voice (Coqui XTTS-v2).
  5. Mux the cloned audio onto the chunk video (video copied untouched).

Usage:
  python clone_chunk.py INPUT.mp4 --src hi --target en --out OUT.mp4
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--src", default="hi")           # whisper/argos code
    ap.add_argument("--target", default="en")
    ap.add_argument("--xtts_lang", default="en")     # XTTS output language
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="large-v3")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        log(f"ERROR: not found {inp}"); sys.exit(1)
    out = Path(args.out) if args.out else inp.with_name(f"{inp.stem} - CLONED{inp.suffix}")
    work = Path("clone_work"); work.mkdir(exist_ok=True)

    # 1. reference audio (clean wav, 22.05k mono is fine for XTTS speaker ref)
    ref = work / "ref.wav"
    log("Extracting reference voice audio...")
    run(["ffmpeg","-y","-i",str(inp),"-vn","-ac","1","-ar","22050",str(ref)])

    # 2. transcribe
    log(f"Transcribing ({args.model})...")
    from faster_whisper import WhisperModel
    m = WhisperModel(args.model, device="cpu", compute_type="int8")
    segs,_ = m.transcribe(str(ref), language=args.src, vad_filter=True, beam_size=5)
    src_text = " ".join(s.text.strip() for s in segs).strip()
    log(f"Source text: {src_text[:200]}")
    if not src_text:
        log("No speech detected; copying original audio."); 
        run(["ffmpeg","-y","-i",str(inp),"-c","copy",str(out)]); 
        print(f"OUTPUT={out}"); return

    # 3. translate with NLLB-200 (much better than offline argos for Hindi->English)
    log("Translating (NLLB-200)...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    NLLB = {"hi": "hin_Deva", "en": "eng_Latn", "bn":"ben_Beng", "ta":"tam_Taml",
            "te":"tel_Telu", "mr":"mar_Deva", "gu":"guj_Gujr", "kn":"kan_Knda",
            "ml":"mal_Mlym", "pa":"pan_Guru", "or":"ory_Orya"}
    src_code = NLLB.get(args.src, "hin_Deva")
    tgt_code = NLLB.get(args.target, "eng_Latn")
    tok_nllb = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    # translate sentence by sentence to keep quality on long text
    import re as _re
    parts = [p.strip() for p in _re.split(r"(?<=[.!?।])\s+", src_text) if p.strip()]
    out_parts = []
    for p in parts:
        tok_nllb.src_lang = src_code
        enc = tok_nllb(p, return_tensors="pt", truncation=True, max_length=512)
        gen = mdl.generate(**enc,
            forced_bos_token_id=tok_nllb.convert_tokens_to_ids(tgt_code),
            max_length=512)
        out_parts.append(tok_nllb.batch_decode(gen, skip_special_tokens=True)[0])
    tgt_text = " ".join(out_parts).strip()
    log(f"Translated: {tgt_text[:200]}")

    # 4. clone voice with XTTS-v2
    log("Generating cloned voice (XTTS-v2)...")
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    cloned = work / "cloned.wav"
    tts.tts_to_file(text=tgt_text, speaker_wav=str(ref),
                    language=args.xtts_lang, file_path=str(cloned))

    # 5. mux cloned audio onto video (video copied; -shortest to align)
    log("Muxing cloned audio onto video...")
    run(["ffmpeg","-y","-i",str(inp),"-i",str(cloned),
         "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k",
         "-shortest",str(out)])
    log(f"DONE -> {out.name}")
    print(f"OUTPUT={out}")

if __name__ == "__main__":
    main()