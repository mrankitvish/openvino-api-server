import sys
import asyncio
import time
import json
import os
import argparse
from typing import List, Optional, Union, Dict, Any, AsyncIterator

import openvino_genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi import status

model_dir = "./models"
device = "CPU"

def validate_model_path(model_dir):
    """Validate that the model directory exists and contains required files."""
    valid_model_paths = []
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=400, detail=f"Model directory does not exist: {model_dir}")

    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # Check for required files
        if not os.path.exists(os.path.join(model_path, "openvino_model.bin")):
            continue
        if not os.path.exists(os.path.join(model_path, "openvino_tokenizer.bin")):
            continue
        if not os.path.exists(os.path.join(model_path, "openvino_detokenizer.bin")):
            continue

        # Check for tokenizer_config.json and validate chat_template
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            continue

        try:
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)

            if "chat_template" not in tokenizer_config or not tokenizer_config["chat_template"]:
                continue
        except json.JSONDecodeError:
            continue
        except Exception as e:
            continue

        valid_model_paths.append(model_path)

    if not valid_model_paths:
        raise HTTPException(status_code=400, detail=f"No valid models found in {model_dir}")

    return valid_model_paths

app = FastAPI(title="OpenAI Compatible API Server")

# Define request and response models to match OpenAI's API structure
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 900
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None

class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

# Custom streaming handler for OpenVINO GenAI
class APIStreamer:
    def __init__(self, response_queue):
        self.response_queue = response_queue
        self.full_response = ""
    
    def __call__(self, subword):
        self.full_response += subword
        self.response_queue.put_nowait(subword)
        # Return False to continue generation
        return False

# Stream generator for streaming responses
async def generate_stream(request: ChatCompletionRequest, model_name: str) -> AsyncIterator[str]:
    # Generate a response ID
    response_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # Send the first chunk with role
    first_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created_time,
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )
        ]
    )
    yield f"data: {json.dumps(first_chunk.dict())}\n\n"

    # Create a queue for the streamer
    queue = asyncio.Queue()
    streamer = APIStreamer(queue)

    # Run the model generation in a separate task
    generation_task = asyncio.create_task(generate_async(request.messages, request.max_tokens, streamer))

    # Stream the output tokens
    try:
        while True:
            try:
                # Get the next token with a timeout
                token = await asyncio.wait_for(queue.get(), timeout=0.1)

                chunk = ChatCompletionStreamResponse(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=token),
                            finish_reason=None
                        )
                    ]
                )

                yield f"data: {json.dumps(chunk.dict())}\n\n"

                # Mark the task as done
                queue.task_done()

            except asyncio.TimeoutError:
                # Check if generation is complete
                if generation_task.done():
                    break

    finally:
        # Send the final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created_time,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=""),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {json.dumps(final_chunk.dict())}\n\n"

        # End the stream
        yield "data: [DONE]\n\n"

async def generate_async(messages, max_tokens, streamer):
    """Run the model generation in an asyncio-friendly way"""
    loop = asyncio.get_event_loop()
    
    def _generate():
        history = [{"role": m.role, "content": m.content} for m in messages]
        model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        answer = pipe.generate(model_inputs, max_new_tokens=max_tokens, streamer=streamer)
        return answer
    
    # Run in a thread pool to avoid blocking the event loop
    return await loop.run_in_executor(None, _generate)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        valid_model_paths = validate_model_path(model_dir)

        # Determine the model to use
        model_name = request.model
        model_path = os.path.join(model_dir, model_name)

        # Validate the model path
        if model_path not in valid_model_paths:
            raise HTTPException(status_code=400, detail=f"Model not found: {model_name}")

        # Load the tokenizer and pipeline for the specified model
        tokenizer = openvino_genai.Tokenizer(model_path)
        pipe = openvino_genai.LLMPipeline(model_path, tokenizer=tokenizer, device=device)

        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                generate_stream(request, model_name),
                media_type="text/event-stream"
            )

        # Non-streaming response
        history = [{"role": m.role, "content": m.content} for m in request.messages]
        model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True)

        # Count tokens for usage metrics (approximation)
        input_tokens = len(model_inputs.split())

        # Generate without streaming
        response_text = pipe.generate(model_inputs, max_new_tokens=request.max_tokens)

        # Count output tokens (approximation)
        output_tokens = len(response_text.split())

        # Create the response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str
    permission: List[Dict[str, Any]]

@app.get("/model")
async def list_models():
    """
    Lists the currently available models.
    """
    model_dir = "./models"
    models = []
    for model_name in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, model_name)):
            models.append(
                Model(
                    id=model_name,
                    object="model",
                    created=int(time.time()),
                    owned_by="openvino",
                    permission=[
                        {
                            "id": "modelperm-xxxxxxxx",
                            "object": "model_permission",
                            "allow_create_engine": True,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": False,
                            "allow_view": False,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                )
            )
    return models


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Provides a health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port="8000", reload=True)
