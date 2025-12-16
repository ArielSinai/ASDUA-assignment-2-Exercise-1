import json
import logging
import os
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error

SERVICE_NAME = os.getenv("SERVICE_NAME", "results-service")


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "therapy-sessions")

logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Results Service")

redis_client: redis.Redis | None = None
minio_client: Minio | None = None


def _make_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


@app.on_event("startup")
async def startup() -> None:
    global redis_client, minio_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Connected to Redis")

    minio_client = _make_minio()
    logger.info("Connected to MinIO")


def _video_id_from_transcript_object(transcript_object: str) -> str:
    return transcript_object.rsplit(".", 1)[0]


async def _find_session_key_by_video_id(video_id: str) -> str | None:
    """
    We store: session_analysis:<bucket>:<uuid>.txt
    We'll scan for: session_analysis:*:<uuid>.txt
    """
    assert redis_client is not None
    pattern = f"session_analysis:*:{video_id}.txt"

    
    cursor = 0
    while True:
        cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=100)
        if keys:
            return keys[0]
        if cursor == 0:
            break
    return None


@app.get("/videos")
async def list_videos(limit: int = 50) -> dict[str, Any]:
    """
    List analyzed videos stored in Redis under session_analysis:* keys.
    """
    assert redis_client is not None

    results: list[dict[str, Any]] = []

    cursor = 0
    
    while True:
        cursor, keys = await redis_client.scan(
            cursor=cursor, match="session_analysis:*", count=200
        )

        for key in keys:
            raw = await redis_client.get(key)
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                continue

            transcript_object = doc.get("transcript_object", "")
            bucket = doc.get("bucket", MINIO_BUCKET)
            video_id = _video_id_from_transcript_object(transcript_object) if transcript_object else ""

            
            analysis_object = f"{video_id}.analysis.json" if video_id else None
            analysis_in_minio = False
            if analysis_object and minio_client is not None:
                try:
                    minio_client.stat_object(bucket, analysis_object)
                    analysis_in_minio = True
                except S3Error:
                    analysis_in_minio = False

            results.append(
                {
                    "video_id": video_id,
                    "bucket": bucket,
                    "transcript_object": transcript_object,
                    "original_filename": doc.get("original_filename"),
                    "created_at": doc.get("created_at"),
                    "analysis_in_minio": analysis_in_minio,
                }
            )

            if len(results) >= limit:
                return {"count": len(results), "items": results}

        if cursor == 0:
            break

    return {"count": len(results), "items": results}


@app.get("/videos/{video_id}")
async def get_video_analysis(video_id: str) -> dict[str, Any]:
    """
    Get full analysis doc from Redis by video_id.
    """
    assert redis_client is not None

    key = await _find_session_key_by_video_id(video_id)
    if not key:
        raise HTTPException(status_code=404, detail="video_id not found")

    raw = await redis_client.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="analysis document missing")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


@app.get("/videos/{video_id}/analysis-file")
async def get_analysis_file(video_id: str) -> dict[str, Any]:
    """
    Fetch the analysis JSON file from MinIO (if exists) and return it.
    """
    if minio_client is None:
        raise HTTPException(status_code=500, detail="MinIO not initialized")

    object_name = f"{video_id}.analysis.json"
    try:
        resp = minio_client.get_object(MINIO_BUCKET, object_name)
        data = resp.read().decode("utf-8")
        return json.loads(data)
    except S3Error:
        raise HTTPException(status_code=404, detail="analysis file not found in MinIO")
    except json.JSONDecodeError:
        return {"raw": data}
