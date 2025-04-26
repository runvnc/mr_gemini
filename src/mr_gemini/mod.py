from lib.providers.services import service
import os
import base64
from io import BytesIO
from openai import AsyncOpenAI
from lib.utils.debug import debug_box

client = AsyncOpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.0, max_tokens=4200, num_gpu_layers=0):
    try:
        print("gemini stream_chat (OpenAI compatible mode)")
        
        if model is None:
            model_name = os.environ.get("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
        else:
            model_name = model

        # Create streaming response using OpenAI compatibility layer
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format= { "type": "json_object" },
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )

        print("Opened stream with model:", model_name)
        
        async def content_stream(original_stream):
            async for chunk in original_stream:
                if os.environ.get('AH_DEBUG') == 'True':
                    print('\033[93m' + str(chunk) + '\033[0m', end='')
                    print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('Gemini (OpenAI mode) error:', e)
        #raise

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
        print('Error getting models (Anthropic):', e)
        return { "stream_chat": [] }

