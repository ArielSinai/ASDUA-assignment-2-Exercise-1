# upload-service/main.py

import io
import json
import logging
import os
import uuid
import asyncio
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from minio import Minio
from minio.error import S3Error
import aio_pika


SERVICE_NAME = os.getenv("SERVICE_NAME", "upload-service")

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

handler = logging.StreamHandler()  # stdout
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)




app = FastAPI(title="Upload Service")

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

    await channel.set_qos(prefetch_count=10)

    queue = await channel.declare_queue(
        RABBITMQ_QUEUE, durable=True
    )
    logger.info(f"RabbitMQ queue '{RABBITMQ_QUEUE}' declared")
    return conn, channel, queue  # type: ignore


async def publish_video_uploaded_message(
    bucket: str, object_name: str, original_filename: str
) -> None:
    """שולח הודעה ל-RabbitMQ על וידאו שהועלה."""
    if rabbit_channel is None:
        logger.error("RabbitMQ channel not initialized – skipping publish")
        return

    message_body: dict[str, Any] = {
        "event": "video_uploaded",
        "bucket": bucket,
        "object_name": object_name,
        "original_filename": original_filename,
    }
    body_bytes = json.dumps(message_body).encode("utf-8")

    message = aio_pika.Message(
        body=body_bytes,
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )

    await rabbit_channel.default_exchange.publish(
        message, routing_key=RABBITMQ_QUEUE
    )
    logger.info(
        f"Published message to RabbitMQ queue '{RABBITMQ_QUEUE}' "
        f"for object '{object_name}'"
    )




@app.on_event("startup")
async def on_startup() -> None:
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
            logger.info("Connected to RabbitMQ successfully")
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"RabbitMQ not ready yet (attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(delay_seconds)

    if rabbit_channel is None:
        logger.error(
            "Giving up connecting to RabbitMQ – "
            "service will run WITHOUT queue, events won't be published."
        )

    logger.info("Startup complete")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global rabbit_conn

    logger.info(f"{SERVICE_NAME} shutting down...")
    if rabbit_conn is not None:
        await rabbit_conn.close()
        logger.info("RabbitMQ connection closed")




@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    if minio_client is None:
        logger.error("MinIO client is not initialized")
        raise HTTPException(status_code=500, detail="Storage not available")

    
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video (content-type video/*)",
        )

    extension = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "mp4"
    object_name = f"{uuid.uuid4().hex}.{extension}"

    try:
        logger.info(
            f"Receiving upload: original='{file.filename}', "
            f"stored_as='{object_name}'"
        )
        contents = await file.read()
        data_stream = io.BytesIO(contents)

        
        minio_client.put_object(
            MINIO_BUCKET,
            object_name,
            data_stream,
            length=len(contents),
            content_type=file.content_type,
        )
        logger.info(
            f"Saved video to MinIO: bucket='{MINIO_BUCKET}', object='{object_name}'"
        )

        
        await publish_video_uploaded_message(
            bucket=MINIO_BUCKET,
            object_name=object_name,
            original_filename=file.filename,
        )

        return JSONResponse(
            {
                "message": "File uploaded successfully",
                "bucket": MINIO_BUCKET,
                "object_name": object_name,
            }
        )
    except S3Error as e:
        logger.exception(f"MinIO error while saving file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save file to storage",
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Unexpected error while handling upload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error",
        ) from e
