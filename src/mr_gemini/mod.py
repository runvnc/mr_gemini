from lib.providers.services import service
import os
import base64
import asyncio # Added for asyncio.sleep
from io import BytesIO
from openai import AsyncOpenAI
from lib.utils.debug import debug_box

from lib.utils.backoff import ExponentialBackoff

client = AsyncOpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)



# Initialize a global backoff manager for Gemini services
# You might want to make these parameters configurable via environment variables or a config file
gemini_backoff_manager = ExponentialBackoff(initial_delay=2.0, max_delay=120.0, factor=2, jitter=True)
MAX_RETRIES = 3

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.01, max_tokens=20000, num_gpu_layers=0):
    if model is None:
        model_name = os.environ.get("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
    else:
        model_name = model
    
    print(f"Gemini stream_chat (OpenAI compatible mode) for model: {model_name}")

    for attempt_num in range(MAX_RETRIES + 1):
        try:
            # Check and honor backoff before making the API call
            wait_time = gemini_backoff_manager.get_wait_time(model_name)
            if wait_time > 0:
                print(f"Gemini model '{model_name}' is in backoff. Waiting for {wait_time:.2f} seconds before attempt {attempt_num + 1}.")
                await asyncio.sleep(wait_time)

            # Create streaming response using OpenAI compatibility layer
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                reasoning_effort="none", 
                response_format= { "type": "json_object" }, 
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"Opened stream with model: {model_name} (Attempt {attempt_num + 1})")
            
            # If successful, record success and prepare the content stream
            gemini_backoff_manager.record_success(model_name)
           
            async def content_stream(original_stream):
                async for chunk in original_stream:
                    if os.environ.get('AH_DEBUG') == 'True':
                        # print('\033[93m' + str(chunk) + '\033[0m', end='') # Full chunk
                        # print('\033[92m' + str(chunk.choices[0].delta) + '\033[0m', end='') # Delta object
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                    yield chunk.choices[0].delta.content or ""
            return content_stream(stream)

        except Exception as e:
            error_message = str(e).lower()
            is_rate_limit_error = (
                "limit" in error_message or 
                "overloaded" in error_message or 
                "rate" in error_message or 
                "quota" in error_message or 
                "429" in error_message # HTTP 429 Too Many Requests
            )

            if is_rate_limit_error:
                gemini_backoff_manager.record_failure(model_name)
                print(f"Gemini (OpenAI mode) rate limit error for '{model_name}' on attempt {attempt_num + 1}/{MAX_RETRIES + 1}: {e}")
                if attempt_num < MAX_RETRIES:
                    next_wait = gemini_backoff_manager.get_wait_time(model_name)
                    print(f"Will retry after ~{next_wait:.2f} seconds.")
                    continue # Go to the next iteration of the loop to retry
                else:
                    print(f"Max retries ({MAX_RETRIES + 1}) reached for '{model_name}'. Raising final error.")
                    raise e # Max retries exceeded, raise the last error
            else:
                # Not a recognized rate limit error, raise immediately
                print(f"Gemini (OpenAI mode) non-rate-limit error for '{model_name}': {e}")
                raise e


@service()
async def format_image_message(pil_image, context=None):
    """Format image for Gemini using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for Gemini"""
    return 4096, 4096, 16777216  # Max width, height, pixels


@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    try:
        print("....!")
        debug_box("Gemini models:")
        all_models = await client.models.list()
        print(all_models)
        print('=====>', all_models)
        ids = []
        for model in all_models.data:
            print('#####################################################')
            print(model)
            ids.append(model.id)

        return { "stream_chat": ids }
    except Exception as e:
        print('Error getting models (Gemini):', e)
        return { "stream_chat": [] }

