import torch
from transformers import pipeline


MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
VIDEO_LENGTH_LIMIT_S = 3600  # TODO(jimmy) limit to 1 hour video files

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

print(pipe)