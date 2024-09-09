import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

class VercelStreamResponse(StreamingResponse):
    @classmethod
    def convert_text(cls, token: str):
        return f"{token}\n\n"

    def __init__(
        self,
        request: Request,
        response: StreamingAgentChatResponse,
    ):
        content = self.content_generator(request, response)
        super().__init__(content=content, media_type="text/event-stream")

    @classmethod
    async def content_generator(
        cls,
        request: Request,
        response: StreamingAgentChatResponse,
    ):
        try:
            if hasattr(response, 'async_response_gen'):
                async for token in response.async_response_gen():
                    yield cls.convert_text(token)
            elif hasattr(response, 'body_iterator'):
                async for chunk in response.body_iterator:
                    yield cls.convert_text(chunk.decode())
            else:
                yield cls.convert_text(str(response))
        except Exception as e:
            print(f"Error in content_generator: {e}")
        finally:
            if await request.is_disconnected():
                print("Client disconnected")
                return
