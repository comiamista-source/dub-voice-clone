#!/usr/bin/env python3
"""
Sarvam Dubbing engine (cloned voice) - dubs ONE chunk via dashboard.sarvam.ai.
Uses SARVAM_API_KEY. Server-side: transcribe -> translate -> CLONE original
voice -> render. Fast, and matches the original speaker.

Usage: python dub_sarvam.py INPUT.mp4 --out OUT.mp4 --src Hindi --target English
"""
import argparse, os, sys, time, requests

BASE="https://dashboard.sarvam.ai"
API=f"{BASE}/api/dubbing"
KEY=os.environ.get("SARVAM_API_KEY","")
H={"api-subscription-key":KEY}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--out", default=None)
    ap.add_argument("--src", default="Hindi"); ap.add_argument("--target", default="English")
    ap.add_argument("--speakers", type=int, default=1); ap.add_argument("--genre", default="monologue")
    a=ap.parse_args()
    inp=a.input
    if not os.path.exists(inp): log(f"ERROR not found {inp}"); sys.exit(1)
    out=a.out or (os.path.splitext(inp)[0]+" - English DUB.mp4")

    # create
    log("Creating dubbing job...")
    payload={"src_lang":a.src,"target_langs":[a.target],"job_name":os.path.basename(inp),
             "num_speakers":a.speakers,"genre":a.genre,"editor_flow":False}
    r=requests.post(f"{API}/jobs",json=payload,headers=H)
    if r.status_code!=200: log(f"create failed {r.status_code}: {r.text[:300]}"); sys.exit(1)
    d=r.json()["data"]; jid=d["job_id"]; log(f"job {jid}")

    # upload
    log(f"Uploading {os.path.getsize(inp)//1024}KB...")
    with open(inp,"rb") as f:
        ru=requests.put(d["upload_url"],data=f,headers={"x-ms-blob-type":"BlockBlob"})
    if ru.status_code not in (200,201): log(f"upload failed {ru.status_code}"); sys.exit(1)

    # start
    rs=requests.post(f"{API}/jobs/{jid}/start",headers=H)
    if rs.status_code!=200: log(f"start failed {rs.status_code}: {rs.text[:300]}"); sys.exit(1)
    log("processing started")

    # poll
    last=""; t0=time.time()
    while True:
        if time.time()-t0>1800: log("timeout 30min"); sys.exit(1)
        rp=requests.get(f"{API}/jobs/{jid}/live-status",headers=H)
        if rp.status_code!=200: time.sleep(5); continue
        data=rp.json().get("data",{})
        st=data.get("status","unknown"); prog=data.get("progress",0); step=data.get("current_step_label","")
        if step and step!=last: log(f"[{prog:3.0f}%] {step}"); last=step
        if st=="completed":
            exp=data.get("export",{}) or {}
            url=exp.get("dubbed_video_url","")
            if url:
                log("downloading dubbed result...")
                # media-proxy for azure blobs
                if "blob.core.windows.net" in url:
                    durl=f"{BASE}/api/media-proxy?url={requests.utils.quote(url,safe='')}"
                else: durl=url
                dr=requests.get(durl,stream=True,headers=H)
                if dr.status_code!=200: dr=requests.get(url,stream=True)
                with open(out,"wb") as f:
                    for c in dr.iter_content(8192): f.write(c)
                log(f"DONE -> {out}"); print(f"OUTPUT={out}"); return
            log("completed but no url"); sys.exit(1)
        if st in ("failed","partial_failure"):
            log(f"job failed: {data.get('error_message','?')}"); sys.exit(1)
        time.sleep(5)

if __name__=="__main__":
    if not KEY: log("ERROR: SARVAM_API_KEY not set"); sys.exit(2)
    main()