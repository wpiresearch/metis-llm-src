import ollama

async def get_response(message: str, historical_messages: list[dict]) -> str:
    response = ollama.chat(model="llama3.2", messages=historical_messages + [{"role": "user", "content":message}])
    return response.message.content
