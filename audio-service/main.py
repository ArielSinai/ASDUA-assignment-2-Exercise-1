

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Any

import aio_pika
from minio import Minio
from minio.error import S3Error

SERVICE_NAME = os.getenv("SERVICE_NAME", "audio-service")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "therapy-sessions")

RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/"
)
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "video_uploaded")



logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)



minio_client: Minio | None = None
rabbit_conn: aio_pika.RobustConnection | None = None
rabbit_channel: aio_pika.Channel | None = None
rabbit_queue: aio_pika.Queue | None = None



def init_minio() -> Minio:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    if not client.bucket_exists(MINIO_BUCKET):
        logger.info(f"Bucket {MINIO_BUCKET} not found, creating it...")
        client.make_bucket(MINIO_BUCKET)
    else:
        logger.info(f"Bucket {MINIO_BUCKET} already exists")

    return client



async def init_rabbitmq() -> tuple[aio_pika.RobustConnection,
                                   aio_pika.Channel,
                                   aio_pika.Queue]:
    logger.info(f"Connecting to RabbitMQ at {RABBITMQ_URL} ...")
    conn = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=5)

    queue = await channel.declare_queue(
        RABBITMQ_QUEUE, durable=True
    )
    logger.info(f"RabbitMQ queue '{RABBITMQ_QUEUE}' declared (consumer)")
    return conn, channel, queue  # type: ignore



def extract_audio_to_mp3(video_path: str, mp3_path: str) -> None:
    """
    מקבל path של וידאו ו-path ליצירת mp3.
    משתמש ב-ffmpeg שיושב בתוך הקונטיינר.
    """
    cmd = [
        "ffmpeg",
        "-y",            # overwrite
        "-i", video_path,
        "-vn",           # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",
        mp3_path,
    ]
    logger.info(f"Running ffmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  


async def handle_video_message(payload: dict[str, Any]) -> None:
    if minio_client is None:
        logger.error("MinIO client not initialized")
        return

    bucket = payload.get("bucket", MINIO_BUCKET)
    object_name = payload["object_name"]
    original_filename = payload.get("original_filename", object_name)

    logger.info(
        f"Processing video from MinIO: bucket='{bucket}', object='{object_name}'"
    )

    try:
        response = minio_client.get_object(bucket, object_name)
        with tempfile.NamedTemporaryFile(suffix="_video", delete=False) as tmp_video:
            data = response.read()
            tmp_video.write(data)
            video_path = tmp_video.name

        
        base_name = object_name.rsplit(".", 1)[0]
        mp3_object_name = f"{base_name}.mp3"

        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = tmp_mp3.name

        
        extract_audio_to_mp3(video_path, mp3_path)

        
        minio_client.fput_object(
            bucket,
            mp3_object_name,
            mp3_path,
            content_type="audio/mpeg",
        )

        logger.info(
            f"Saved MP3 to MinIO: bucket='{bucket}', object='{mp3_object_name}', "
            f"from original='{original_filename}'"
        )

    except S3Error as e:
        logger.exception(f"MinIO error while processing video: {e}")
    except subprocess.CalledProcessError as e:
        logger.exception(f"ffmpeg failed: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Unexpected error while handling video message: {e}")



async def consume_messages() -> None:
    assert rabbit_channel is not None
    assert rabbit_queue is not None

    async def on_message(message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                body_str = message.body.decode("utf-8")
                payload = json.loads(body_str)
                logger.info(f"Received RabbitMQ message: {payload}")
                await handle_video_message(payload)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Failed to handle message: {e}")

    await rabbit_queue.consume(on_message, no_ack=False)
    logger.info("Started consuming RabbitMQ messages")




async def startup() -> None:
    global minio_client, rabbit_conn, rabbit_channel, rabbit_queue

    logger.info(f"{SERVICE_NAME} starting up...")

    # MinIO
    try:
        minio_client = init_minio()
        logger.info("Connected to MinIO successfully")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to initialize MinIO: {e}")
        raise

    
    max_attempts = int(os.getenv("RABBITMQ_CONNECT_RETRIES", "20"))
    delay_seconds = float(os.getenv("RABBITMQ_CONNECT_DELAY", "5"))

    rabbit_conn = rabbit_channel = rabbit_queue = None  # type: ignore[assignment]

    for attempt in range(1, max_attempts + 1):
        try:
            rabbit_conn, rabbit_channel, rabbit_queue = await init_rabbitmq()
            logger.info("Connected to RabbitMQ successfully (consumer)")
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"RabbitMQ not ready yet (attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay_seconds)

    if rabbit_channel is None or rabbit_queue is None:
        logger.error(
            "Giving up connecting to RabbitMQ – audio-service will run "
            "WITHOUT consuming messages."
        )
        return

    
    asyncio.create_task(consume_messages())
    logger.info("Startup complete, consumer task started.")


async def shutdown() -> None:
    global rabbit_conn
    logger.info(f"{SERVICE_NAME} shutting down...")
    if rabbit_conn is not None:
        await rabbit_conn.close()
        logger.info("RabbitMQ connection closed")



def main() -> None:
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(startup())
        loop.run_forever()
    finally:
        loop.run_until_complete(shutdown())
        loop.close()


if __name__ == "__main__":
    main()
