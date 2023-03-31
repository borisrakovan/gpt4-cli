import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# load environment variables from .env file before importing the GPT client
envfile = Path(os.path.abspath(__file__)).parent / ".env"
if not envfile.exists():
    raise FileNotFoundError(f"{envfile} does not exist")

load_dotenv(envfile)


from client import ChatGPTClient  # noqa: E402

_GPT_CLIENT_SYSTEM_MESSAGE = "You are a helpful technical assistant"


async def main():
    client = ChatGPTClient(_GPT_CLIENT_SYSTEM_MESSAGE)

    while True:
        message = input("You: ")
        if message.startswith(">"):
            filename = message[1:].strip()
            try:
                with open(filename) as f:
                    message = f.read()
            except FileNotFoundError:
                print(f"File {filename} not found")
                continue
            print(message)
        elif message == "quit":
            break
        reply = await client.chat(message)
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
