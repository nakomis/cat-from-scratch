#!/usr/bin/env python3
"""
Local SAM 2 segmentation server for BootBoots cat annotation.

Run once to download the checkpoint:
    python scripts/sam_server.py --download

Then start the server (HTTP, for local dev):
    python scripts/sam_server.py

With TLS (required when accessed from the deployed HTTPS sandbox site):
    bash scripts/fetch_certs.sh          # copies certs from nasbox
    python scripts/sam_server.py --tls

The server listens on 0.0.0.0:7861.  With --tls it serves HTTPS so that
sandbox.nakomis.com (CloudFront HTTPS) can call it without mixed-content errors.
The certificate covers *.nasbox.nakomis.com; point phi.nasbox.nakomis.com at
this machine's LAN IP (172.29.0.26) via a Route53 A record.
"""

import argparse
import base64
import io
import urllib.request
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupt S3 images
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent

CHECKPOINT_DIR = ROOT / "models" / "sam2"
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{CHECKPOINT_NAME}"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

UNCERTAIN_BORDER_PX = 12  # dilation radius for the "maybe cat" border region

app = FastAPI(title="SAM 2 Segmentation Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    global predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = CHECKPOINT_DIR / CHECKPOINT_NAME
    if not checkpoint.exists():
        raise RuntimeError(
            f"SAM 2 checkpoint not found at {checkpoint}.\n"
            f"Run: python scripts/sam_server.py --download"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading SAM 2 ({CHECKPOINT_NAME}) on {device}...")
    model = build_sam2(MODEL_CFG, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    print("SAM 2 ready.")


# ---------------------------------------------------------------------------
# Trimap overlay generation
# ---------------------------------------------------------------------------

def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation via max-pool — no scipy required."""
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    out = torch.nn.functional.max_pool2d(t, kernel, stride=1, padding=radius)
    return out.squeeze().numpy().astype(bool)


def make_trimap_overlay(mask: np.ndarray) -> Image.Image:
    """
    Convert a binary SAM mask to an RGBA overlay using the same colour scheme
    as our Oxford trimap visualisations:
      - foreground  → transparent (original image shows through)
      - uncertain   → light slate grey (border region)
      - background  → semi-transparent grey
    """
    mask = mask.astype(bool)
    dilated = _dilate(mask, UNCERTAIN_BORDER_PX)
    h, w = mask.shape

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[~dilated] = [100, 100, 100, 170]       # background: grey
    overlay[dilated & ~mask] = [180, 190, 200, 220] # uncertain: slate
    # foreground stays (0,0,0,0) — fully transparent

    return Image.fromarray(overlay, "RGBA")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Point(BaseModel):
    x: float
    y: float
    label: int  # 1 = positive (cat), 0 = negative (background)


class SegmentRequest(BaseModel):
    image: str          # base64-encoded image (JPEG or PNG)
    points: list[Point]
    display_width: int  # dimensions of the canvas on-screen
    display_height: int # (points are in display coords; server scales to image coords)


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class SegmentResponse(BaseModel):
    overlay: str        # base64 RGBA PNG overlay at original image resolution
    bounding_box: BoundingBox
    foreground_pixels: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not req.points:
        raise HTTPException(status_code=400, detail="At least one point required.")

    # Decode image
    image_bytes = base64.b64decode(req.image)
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)
    orig_h, orig_w = image_np.shape[:2]

    # Scale display-space points to original image coords
    sx = orig_w / req.display_width
    sy = orig_h / req.display_height
    coords = np.array([[p.x * sx, p.y * sy] for p in req.points], dtype=np.float32)
    labels = np.array([p.label for p in req.points], dtype=np.int32)

    predictor.set_image(image_np)
    masks, _scores, _logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )

    mask = masks[0]  # (H, W) bool

    # Bounding box from foreground pixels
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) and len(cols):
        bbox = BoundingBox(xmin=int(cols[0]), ymin=int(rows[0]),
                           xmax=int(cols[-1]), ymax=int(rows[-1]))
    else:
        bbox = BoundingBox(xmin=0, ymin=0, xmax=orig_w, ymax=orig_h)

    overlay_img = make_trimap_overlay(mask)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return SegmentResponse(
        overlay=overlay_b64,
        bounding_box=bbox,
        foreground_pixels=int(mask.sum()),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def download_checkpoint():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / CHECKPOINT_NAME
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {CHECKPOINT_NAME} (~46 MB)…")
    urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download the SAM 2 checkpoint and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--tls", action="store_true",
                        help="Serve over HTTPS using certs/fullchain.pem + certs/privkey.pem")
    args = parser.parse_args()

    if args.download:
        download_checkpoint()
    else:
        load_model()
        cert_dir = ROOT / "certs"
        ssl_kwargs = {}
        if args.tls:
            ssl_kwargs = {
                "ssl_certfile": str(cert_dir / "fullchain.pem"),
                "ssl_keyfile":  str(cert_dir / "privkey.pem"),
            }
            print("TLS enabled — serving HTTPS")
        uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)
