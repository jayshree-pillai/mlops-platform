import os, json, pickle, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss, numpy as np, yaml
from openai import OpenAI
import boto3

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"

with open(CONFIG_DIR / "base.yml", "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

EMBED_MODEL = _CFG.get("embedding_model", "text-embedding-3-large")
ARTIFACT_ENV = _CFG.get("artifact_dir_env", "ARTIFACT_DIR")
DEFAULT_ARTIFACT_DIR = _CFG.get("default_artifact_dir", "/srv/rag-api/artifacts/faiss/current")

_oa = OpenAI()

_s3 = None
def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3

_S3_RE = re.compile(r"^s3://([^/]+)/(.+)$")

def _parse_s3(uri: str):
    m = _S3_RE.match(uri)
    if not m:
        raise ValueError(f"Bad S3 URI: {uri}")
    return m.group(1), m.group(2).rstrip("/") + "/"

def _ensure_local_from_s3(s3_uri: str) -> Path:
    """
    Mirror the S3 'current/' (or versions/<V>/) folder to a local cache and return that Path.
    Required files: index.faiss, pca.transform, index.pkl, manifest.json
    """
    s3 = _s3_client()
    bucket, key_prefix = _parse_s3(s3_uri)
    # 1) read manifest to get version (for stable cache dir)
    mani_key = key_prefix + "manifest.json"
    mani_obj = s3.get_object(Bucket=bucket, Key=mani_key)
    manifest = json.loads(mani_obj["Body"].read().decode("utf-8"))
    version = manifest.get("version", "unknown")
    cache_dir = Path(f"/tmp/rag-artifacts/{version}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2) download the four files if missing
    need = ["index.faiss", "pca.transform", "index.pkl", "manifest.json"]
    for name in need:
        dest = cache_dir / name
        if not dest.exists():
            s3.download_file(bucket, key_prefix + name, str(dest))

    # also write version marker
    (cache_dir / "version.txt").write_text(version, encoding="utf-8")
    return cache_dir

def _resolve_artifact_dir() -> Path:
    """Return a local directory path. If env points to s3://..., cache locally."""
    val = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    if str(val).startswith("s3://"):
        return _ensure_local_from_s3(val)
    p = Path(val)
    if not p.exists():
        raise FileNotFoundError(f"ARTIFACT_DIR not found: {p}")
    return p

def _read_manifest(ad: Path) -> Dict[str, Any]:
    with (ad / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_index(ad: Path):
    return faiss.read_index(str(ad / "index.faiss"))

def _read_pca(ad: Path):
    return faiss.read_VectorTransform(str(ad / "pca.transform"))

def _read_meta(ad: Path) -> Dict[str, Any]:
    with (ad / "index.pkl").open("rb") as f:
        return pickle.load(f)

def _embed(texts: List[str]) -> np.ndarray:
    resp = _oa.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = np.vstack([np.asarray(e.embedding, dtype=np.float32) for e in resp.data])
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

class Retriever:
    def __init__(self, k_default: int = 4):
        self.k_default = k_default
        self._loaded = False
        self._ad = None
        self._idx = None
        self._pca = None
        self._texts = None
        self._metas = None
        self.manifest = None
        self.version = None

    def ensure_loaded(self):
        if self._loaded:
            return
        ad = _resolve_artifact_dir()
        self.manifest = _read_manifest(ad)
        self.version = self.manifest.get("version", "unknown")
        self._idx = _read_index(ad)
        self._pca = _read_pca(ad)
        meta = _read_meta(ad)
        self._texts = meta.get("texts", [])
        self._metas = meta.get("metadata", [{}] * len(self._texts))
        self._ad = ad
        self._loaded = True

    def retrieve(self, query: str, k: int = None):
        self.ensure_loaded()
        k = k or self.k_default
        qv = _embed([query])
        qv = self._pca.apply_py(qv)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        scores, idxs = self._idx.search(qv, k)
        idxs, scores = idxs[0].tolist(), scores[0].tolist()
        out = []
        for i, s in zip(idxs, scores):
            if 0 <= i < len(self._texts):
                out.append((self._texts[i], self._metas[i], float(s)))
        return out

    def info(self) -> Dict[str, Any]:
        self.ensure_loaded()
        return {
            "version": self.version,
            "artifact_dir": str(self._ad),
            "ntotal": int(self._idx.ntotal),
            "pca_outdim": int(self._idx.d),
            "embedding_model": EMBED_MODEL,
        }

_singleton: Retriever = None
def get_retriever(k_default: int = None) -> Retriever:
    global _singleton
    if _singleton is None:
        _singleton = Retriever(k_default=k_default or _CFG.get("default_top_k", 4))
        _singleton.ensure_loaded()
    return _singleton
