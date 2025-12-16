import asyncio
import json
import logging
import os
import hashlib
from datetime import datetime
from io import BytesIO
from typing import Any

import aio_pika
from minio import Minio
from minio.error import S3Error
import redis.asyncio as redis
from openai import OpenAI



SERVICE_NAME = os.getenv("SERVICE_NAME", "analysis-service")


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "therapy-sessions")


RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
TRANSCRIPT_READY_QUEUE = os.getenv("TRANSCRIPT_READY_QUEUE", "transcript_ready")


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")



logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)



minio_client: Minio | None = None
rabbit_conn: aio_pika.RobustConnection | None = None
rabbit_channel: aio_pika.Channel | None = None
redis_client: redis.Redis | None = None
openai_client = OpenAI(api_key=OPENAI_API_KEY)




def init_minio() -> Minio:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    logger.info("Connected to MinIO")
    return client


async def init_rabbitmq(url: str, queue_name: str):
    retries = 10
    delay = 3

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"Connecting to RabbitMQ at {url} "
                f"(attempt {attempt}/{retries})..."
            )
            connection = await aio_pika.connect_robust(url)
            channel = await connection.channel()
            queue = await channel.declare_queue(queue_name, durable=True)
            logger.info("Connected to RabbitMQ successfully")
            return connection, channel, queue
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RabbitMQ not ready yet: {e}")
            if attempt == retries:
                logger.error("Giving up connecting to RabbitMQ – exiting.")
                raise
            await asyncio.sleep(delay)


async def init_redis(url: str) -> redis.Redis:
    client: redis.Redis = redis.from_url(url, decode_responses=True)
    
    await client.ping()
    logger.info("Connected to Redis")
    return client


# LLM logic

EMOTIONS = [
    "happy",
    "joyful",
    "calm",
    "neutral",
    "sad",
    "angry",
    "anxious",
    "fearful",
    "guilty",
    "ashamed",
    "frustrated",
    "overwhelmed",
    "lonely",
    "hopeful",
]

TOPIC_HINT = [
    "mother", "father", "parents", "family", "friends", "partner",
    "relationship", "work", "school", "health", "self-esteem",
    "boundaries", "goals", "trauma", "grief", "childhood",
]


def build_prompt(transcript_text: str) -> str:
    emotions_str = ", ".join(EMOTIONS)
    topics_str = ", ".join(TOPIC_HINT)

    return f"""
You are a helpful clinical conversation analyzer.

The input is a transcript of a therapy session. Each line looks like:
"Speaker 0: some text"
"Speaker 1: some text"
etc.

Exactly one of the speakers is the THERAPIST. The others (usually one) are PATIENTS.

Your tasks:
1. Infer which Speaker ID is the therapist and which is the patient.
2. For each utterance (each line), create a JSON object with:
   - "speaker_id": the numeric id from the transcript (0, 1, ...)
   - "speaker_role": "therapist" or "patient"
   - "text": the original sentence text
   - "topic": a short topic label (you may use free text, but prefer: {topics_str})
   - "emotion": one of [{emotions_str}]
3. Also provide a high-level summary:
   - overall_emotion
   - positive_topics: list of topics that make the patient feel good
   - negative_topics: list of topics that make the patient feel bad
   - notes_for_therapist: short bullet points with suggestions for next session.

Return ONLY valid JSON.
Do not wrap with ``` or any extra text.
Return STRICTLY a single JSON object with this structure:

{{
  "speakers": {{
    "therapist_id": <int>,
    "patient_ids": [<int>, ...]
  }},
  "utterances": [
    {{
      "speaker_id": <int>,
      "speaker_role": "therapist" | "patient",
      "text": "<original text>",
      "topic": "<topic>",
      "emotion": "<emotion>",
    }},
    ...
  ],
  "summary": {{
    "overall_emotion": "<emotion>",
    "positive_topics": ["...", "..."],
    "negative_topics": ["...", "..."],
    "notes_for_therapist": ["...", "..."]
  }}
}}

Transcript:
{transcript_text}
""".strip()


async def analyze_with_llm(
    transcript_text: str,
) -> dict[str, Any]:
    """
    Analyze transcript with OpenAI, with Redis caching.
    """

    assert redis_client is not None

    # Cache key based on SHA256 of the transcript
    digest = hashlib.sha256(transcript_text.encode("utf-8")).hexdigest()
    cache_key = f"analysis_cache:{digest}"

    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Using cached analysis from Redis")
        return json.loads(cached)

    prompt = build_prompt(transcript_text)

    logger.info("Calling OpenAI for analysis...")
    
    loop = asyncio.get_running_loop()

    def _call_openai() -> str:
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        return resp.output_text  

    output_text: str = await loop.run_in_executor(None, _call_openai)
    logger.info("Received response from OpenAI")

    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        logger.error(
            "Failed to parse LLM output as JSON. "
            "Storing raw text under 'raw_output'."
        )
        data = {"raw_output": output_text}

    await redis_client.set(cache_key, json.dumps(data))
    logger.info("Stored analysis in Redis cache")
    return data





async def handle_transcript_message(payload: dict[str, Any]) -> None:
    """
    Handle one 'transcript_ready' message:
    - download .txt from MinIO
    - run LLM analysis (with caching)
    - save result in Redis and in MinIO as .analysis.json
    """
    assert minio_client is not None
    assert redis_client is not None

    bucket = payload.get("bucket", MINIO_BUCKET)
    object_name = payload["object_name"]
    original_filename = payload.get("original_filename", object_name)

    logger.info(
        f"Handling transcript from MinIO: "
        f"bucket='{bucket}', object='{object_name}'"
    )

    try:
        # Download .txt from MinIO
        response = minio_client.get_object(bucket, object_name)
        transcript_bytes = response.read()
        transcript_text = transcript_bytes.decode("utf-8")

        # Run analysis (with caching)
        analysis = await analyze_with_llm(transcript_text)

        # Save full analysis in Redis as "database"
        session_key = f"session_analysis:{bucket}:{object_name}"
        doc = {
            "bucket": bucket,
            "transcript_object": object_name,
            "original_filename": original_filename,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "analysis": analysis,
        }
        await redis_client.set(session_key, json.dumps(doc))
        logger.info(f"Saved analysis document in Redis under '{session_key}'")

        # Save analysis to MinIO as JSON file next to the transcript
        base = object_name.rsplit(".", 1)[0]
        analysis_name = f"{base}.analysis.json"
        content_bytes = json.dumps(analysis, indent=2).encode("utf-8")
        stream = BytesIO(content_bytes)

        minio_client.put_object(
            bucket,
            analysis_name,
            stream,
            length=len(content_bytes),
            content_type="application/json",
        )
        logger.info(
            f"Saved analysis file to MinIO: "
            f"bucket='{bucket}', object='{analysis_name}'"
        )

    except S3Error as e:
        logger.exception(f"MinIO error while handling transcript: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Unexpected error in analysis-service: {e}")





async def main() -> None:
    global minio_client, rabbit_conn, rabbit_channel, redis_client

    logger.info(f"{SERVICE_NAME} starting up...")
    minio_client = init_minio()
    redis_client = await init_redis(REDIS_URL)
    rabbit_conn, rabbit_channel, queue = await init_rabbitmq(
        RABBITMQ_URL, TRANSCRIPT_READY_QUEUE
    )

    async def on_message(message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                payload = json.loads(message.body.decode("utf-8"))
                logger.info(f"Received message: {payload}")
                await handle_transcript_message(payload)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Failed to process message: {e}")

    await queue.consume(on_message)
    logger.info("Waiting for 'transcript_ready' messages...")
    await asyncio.Future()  


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down analysis-service...")
        if rabbit_conn is not None:
            asyncio.run(rabbit_conn.close())
