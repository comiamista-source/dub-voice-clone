#!/usr/bin/env python3
"""
Voice-clone dubber for ONE chunk - Sarvam text + XTTS voice cloning.
  1. Extract reference audio (speaker's voice) from the chunk.
  2. Sarvam saaras:v3 -> clean English text (transcribe+translate, <=28s segments).
  3. XTTS-v2 -> English speech CLONED from the reference voice.
  4. Mux cloned audio onto the chunk video.

Needs env SARVAM_API_KEY. Usage:
  python clone_chunk.py INPUT.mp4 --out OUT.mp4
"""
import argparse, os, subprocess, sys, time, json, base64, urllib.request

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def dur(p):
    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True)
    try: return float(r.stdout.strip())
    except: return 0.0

def sarvam_stt_translate(wav, key):
    """saaras:v3 transcribe+translate, split into <=28s segments (30s API limit)."""
    total=dur(wav); seg=28.0; texts=[]; i=0; start=0.0
    while start < total:
        part=f"_stt_{i}.wav"
        subprocess.run(["ffmpeg","-y","-ss",str(start),"-t",str(seg),"-i",wav,"-ac","1","-ar","16000",part],capture_output=True)
        if dur(part)>0.3:
            r=subprocess.run(["curl","-s","-X","POST","https://api.sarvam.ai/speech-to-text-translate",
                "-H",f"api-subscription-key: {key}",
                "-F",f"file=@{part};type=audio/wav","-F","model=saaras:v3"],capture_output=True,text=True)
            d=json.loads(r.stdout)
            if "transcript" not in d:
                raise RuntimeError(d.get("error",{}).get("message",str(d)))
            texts.append(d["transcript"].strip())
        start+=seg; i+=1
    return " ".join(texts).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--out", default=None)
    ap.add_argument("--xtts_lang", default="en")
    args=ap.parse_args()

    key=os.environ.get("SARVAM_API_KEY")
    if not key:
        log("ERROR: SARVAM_API_KEY not set"); sys.exit(2)

    inp=args.input
    if not os.path.exists(inp): log(f"ERROR: not found {inp}"); sys.exit(1)
    out=args.out or (os.path.splitext(inp)[0] + " - English DUB.mp4")

    # 1. reference voice
    ref="_ref.wav"
    log("Extracting reference voice audio...")
    subprocess.run(["ffmpeg","-y","-i",inp,"-vn","-ac","1","-ar","22050",ref],capture_output=True)

    # 2. Sarvam clean English text
    log("Transcribe+translate via Sarvam saaras:v3...")
    text=sarvam_stt_translate(ref, key)
    log(f"EN text: {text[:200]}")
    if not text:
        log("No text; copying original audio."); subprocess.run(["ffmpeg","-y","-i",inp,"-c","copy",out],capture_output=True)
        print(f"OUTPUT={out}"); return

    # 3. XTTS clone voice (speaks Sarvam's clean English in the speaker's voice)
    log("Generating cloned voice (XTTS-v2)...")
    from TTS.api import TTS
    tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    cloned="_cloned.wav"
    tts.tts_to_file(text=text, speaker_wav=ref, language=args.xtts_lang, file_path=cloned)

    # 4. mux
    log("Muxing cloned audio onto video...")
    subprocess.run(["ffmpeg","-y","-i",inp,"-i",cloned,"-map","0:v:0","-map","1:a:0",
        "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",out],capture_output=True)
    log(f"DONE -> {out}")
    print(f"OUTPUT={out}")

if __name__=="__main__":
    main()