#!/usr/bin/env python3
"""
F5-TTS voice-clone dubber for ONE chunk (GitHub runner, CPU).
  1. Sarvam saaras:v3 -> clean English text (<=28s segments).
  2. F5-TTS clones the FIXED reference voice (ref_voice.mp3) speaking that English.
  3. Mux onto the chunk video.
Needs SARVAM_API_KEY. Usage: python f5_chunk.py INPUT.mp4 --out OUT.mp4
"""
import argparse, os, subprocess, sys, time, json
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def dur(p):
    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True)
    try: return float(r.stdout.strip())
    except: return 0.0

def sarvam(wav, key):
    r=subprocess.run(["curl","-s","-X","POST","https://api.sarvam.ai/speech-to-text-translate",
        "-H",f"api-subscription-key: {key}","-F",f"file=@{wav};type=audio/wav","-F","model=saaras:v3"],capture_output=True,text=True)
    try: d=json.loads(r.stdout)
    except: return ""
    if "transcript" not in d: raise RuntimeError(d.get("error",{}).get("message",str(d)))
    return d["transcript"].strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--out",default=None)
    a=ap.parse_args()
    key=os.environ.get("SARVAM_API_KEY","")
    if not key: log("no SARVAM_API_KEY"); sys.exit(2)
    inp=a.input; out=a.out or (os.path.splitext(inp)[0]+" - English DUB.mp4")
    here=os.path.dirname(os.path.abspath(__file__))
    refsrc=os.path.join(here,"ref_voice.mp3")

    # reference voice wav (24k for F5)
    ref="_ref.wav"
    subprocess.run(["ffmpeg","-y","-i",refsrc if os.path.exists(refsrc) else inp,"-vn","-ac","1","-ar","24000",ref],capture_output=True)
    log(f"Reference voice: {'ref_voice.mp3' if os.path.exists(refsrc) else 'chunk audio'}")

    # full audio -> 28s STT segments -> collect English text
    full="_full.wav"; subprocess.run(["ffmpeg","-y","-i",inp,"-vn","-ac","1","-ar","16000",full],capture_output=True)
    total=dur(full); texts=[]; start=0.0; i=0
    while start<total:
        s=f"_s{i}.wav"; subprocess.run(["ffmpeg","-y","-ss",str(start),"-t","28",full,"-i",full],capture_output=True) if False else \
        subprocess.run(["ffmpeg","-y","-ss",str(start),"-t","28","-i",full,s],capture_output=True)
        if dur(s)>0.3:
            t=sarvam(s,key);  texts.append(t); log(f"  seg {i}: {t[:80]}")
        start+=28; i+=1
    text=" ".join(texts).strip()
    if not text: log("no text; copying original"); subprocess.run(["ffmpeg","-y","-i",inp,"-c","copy",out]); print(f"OUTPUT={out}"); return

    # F5-TTS clone
    log("Loading F5-TTS (CPU - slow)...")
    from f5_tts.api import F5TTS
    f5=F5TTS()
    cloned="_cloned.wav"
    log("Generating cloned voice...")
    f5.infer(ref_file=ref, ref_text="", gen_text=text, file_wave=cloned)

    log("Muxing...")
    subprocess.run(["ffmpeg","-y","-i",inp,"-i",cloned,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-shortest",out],capture_output=True)
    log(f"DONE -> {out}"); print(f"OUTPUT={out}")

if __name__=="__main__": main()