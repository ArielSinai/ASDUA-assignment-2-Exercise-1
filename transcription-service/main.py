import asyncio
import json
import logging
import os
import tempfile
from io import BytesIO
from typing import Any

import aio_pika
from aio_pika.abc import AbstractRobustConnection, AbstractChannel
from aio_pika import Queue, IncomingMessage
import httpx
from minio import Minio
from minio.error import S3Error


SERVICE_NAME = os.getenv("SERVICE_NAME", "transcription-service")

# MinIO config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "therapy-sessions")

# RabbitMQ config
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
AUDIO_READY_QUEUE = os.getenv("AUDIO_READY_QUEUE", "audio_ready")

# AssemblyAI config
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY env var is required")


# ---------- logging ----------

logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------- globals ----------

minio_client: Minio | None = None
rabbit_conn: AbstractRobustConnection | None = None
rabbit_channel: AbstractChannel | None = None


# ---------- MinIO ----------

def init_minio() -> Minio:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    logger.info("Connected to MinIO")
    return client


# ---------- RabbitMQ ----------

async def init_rabbitmq(
    url: str,
    queue_name: str,
    retries: int = 15,
    delay_seconds: int = 4,
) -> tuple[AbstractRobustConnection, AbstractChannel, Queue]:
    """
    מנסה להתחבר ל־RabbitMQ כמה פעמים, עם הפסקה בין ניסיונות.
    אם אחרי כל הניסיונות עדיין אין חיבור – מרים Exception.
    """
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"Connecting to RabbitMQ at {url} "
                f"(attempt {attempt}/{retries}) ..."
            )
            connection = await aio_pika.connect_robust(url)
            channel = await connection.channel()
            queue = await channel.declare_queue(queue_name, durable=True)
            logger.info("Connected to RabbitMQ successfully")
            return connection, channel, queue

        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning(
                f"RabbitMQ not ready yet: {e!r} "
                f"(attempt {attempt}/{retries})"
            )
            if attempt < retries:
                await asyncio.sleep(delay_seconds)

    # אם הגענו לכאן – כל הניסיונות נכשלו
    logger.error(
        f"Giving up connecting to RabbitMQ after {retries} attempts. "
        f"Last error: {last_exc!r}"
    )
    raise last_exc if last_exc is not None else RuntimeError(
        "Failed to connect to RabbitMQ"
    )


# ---------- AssemblyAI helpers ----------

async def upload_to_assemblyai(file_path: str) -> str:
    """Upload the MP3 file to AssemblyAI and return an upload_url (sync httpx in thread)."""

    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "transfer-encoding": "chunked",
    }

    def _do_upload() -> str:
        with httpx.Client(timeout=None) as client:
            with open(file_path, "rb") as f:
                resp = client.post(
                    ASSEMBLYAI_UPLOAD_URL,
                    headers=headers,
                    content=f,
                )
            resp.raise_for_status()
            data = resp.json()
            return data["upload_url"]

    logger.info("Uploading audio file to AssemblyAI (sync in thread)...")
    upload_url = await asyncio.to_thread(_do_upload)
    logger.info("Upload completed, got upload_url from AssemblyAI")
    return upload_url


async def start_transcription(upload_url: str) -> str:
    """Create a transcription request with speaker diarization enabled (sync httpx in thread)."""

    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }

    body = {
        "audio_url": upload_url,
        "speaker_labels": True,
    }

    def _do_start() -> str:
        with httpx.Client(timeout=None) as client:
            resp = client.post(ASSEMBLYAI_TRANSCRIPT_URL, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()["id"]

    logger.info("Starting transcription with speaker diarization (sync in thread)...")
    transcript_id = await asyncio.to_thread(_do_start)
    logger.info(f"Transcription started. id={transcript_id}")
    return transcript_id


async def wait_for_transcription(transcript_id: str) -> dict[str, Any]:
    """Poll AssemblyAI until the transcription is completed (sync httpx in thread)."""

    headers = {"authorization": ASSEMBLYAI_API_KEY}
    url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"

    def _poll() -> dict[str, Any]:
        with httpx.Client(timeout=None) as client:
            while True:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                status = data["status"]

                if status == "completed":
                    return data
                if status == "error":
                    raise RuntimeError(data.get("error", "Unknown error"))

                # שימי לב: זה בתוך thread, אז אין בעיה עם sleep רגיל
                import time
                time.sleep(3)

    logger.info("Waiting for transcription to complete (sync in thread)...")
    result = await asyncio.to_thread(_poll)
    logger.info("Transcription completed")
    return result


def build_diarized_text(result: dict[str, Any]) -> str:
    """Build a human-readable diarized transcript from AssemblyAI result."""
    utterances = result.get("utterances", [])
    if not utterances:
        # fallback – no diarization, just full text
        return result.get("text", "")

    lines: list[str] = []
    for utt in utterances:
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        lines.append(f"Speaker {speaker}: {text}")
    return "\n".join(lines)


# ---------- main message handler ----------

async def handle_audio_message(payload: dict[str, Any]) -> None:
    """Handle one 'audio_ready' message from RabbitMQ."""
    global minio_client

    if minio_client is None:
        logger.error("MinIO client not initialized")
        return

    bucket = payload.get("bucket", MINIO_BUCKET)
    object_name = payload["object_name"]
    original_filename = payload.get("original_filename", object_name)

    logger.info(
        f"Handling audio file from MinIO: bucket='{bucket}', "
        f"object='{object_name}'"
    )

    try:
        # 1. Download MP3 from MinIO to a temp file
        response = minio_client.get_object(bucket, object_name)
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as tmp_file:
                data = response.read()
                tmp_file.write(data)
                mp3_path = tmp_file.name
        finally:
            response.close()
            response.release_conn()

        # 2. Upload to AssemblyAI
        upload_url = await upload_to_assemblyai(mp3_path)

        # 3. Start transcription with diarization
        transcript_id = await start_transcription(upload_url)

        # 4. Wait for completion
        result = await wait_for_transcription(transcript_id)

        # 5. Build diarized text
        text = build_diarized_text(result)
        if not text:
            logger.warning("Empty transcription text")
            return

        # 6. Save transcript to MinIO as .txt next to the mp3
        base_name = object_name.rsplit(".", 1)[0]
        transcript_name = f"{base_name}.txt"

        content = text.encode("utf-8")
        data_stream = BytesIO(content)

        minio_client.put_object(
            bucket,
            transcript_name,
            data_stream,
            length=len(content),
            content_type="text/plain; charset=utf-8",
        )

        logger.info(
            f"Saved transcript to MinIO: bucket='{bucket}', "
            f"object='{transcript_name}', "
            f"from original='{original_filename}'"
        )

    except S3Error as e:
        logger.exception(f"MinIO error while handling audio: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Unexpected error in transcription-service: {e}")


# ---------- main loop ----------

async def main() -> None:
    global minio_client, rabbit_conn, rabbit_channel

    logger.info(f"{SERVICE_NAME} starting up...")

    # MinIO
    minio_client = init_minio()

    # RabbitMQ – עם יותר ניסיונות וזמן המתנה
    rabbit_conn, rabbit_channel, queue = await init_rabbitmq(
        RABBITMQ_URL,
        AUDIO_READY_QUEUE,
        retries=15,        # למשל 15 ניסיונות
        delay_seconds=4,   # הפרש של 4 שניות ביניהם
    )

    async def on_message(message: IncomingMessage) -> None:
        async with message.process():
            try:
                payload = json.loads(message.body.decode("utf-8"))
                logger.info(f"Received message: {payload}")
                await handle_audio_message(payload)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Failed to process message: {e}")

    await queue.consume(on_message)
    logger.info("Waiting for 'audio_ready' messages...")
    # keep running forever
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down transcription-service...")
        if rabbit_conn is not None:
            asyncio.run(rabbit_conn.close())
