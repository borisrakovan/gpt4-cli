import os
from typing import Any, Literal, TypeAlias

import openai
import tiktoken
from openai.error import AuthenticationError

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIAuthError(Exception):
    pass


AnyDict: TypeAlias = dict[str, Any]

ModelStr = Literal["gpt-4"] | Literal["gpt-4-32k"]

_MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}

_MESSAGE_HISTORY_TRUNCATION_THRESHOLD = 0.9
"""When the message history is too long, truncate it to this fraction of the maximum allowed length."""


class ChatGPTClient:
    def __init__(self, system_message: str, model: ModelStr = "gpt-4", temperature: float = 0.5):
        self.model = model
        self.messages: list[AnyDict] = [{"role": "system", "content": system_message}]
        self.temperature = temperature

    async def chat(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        reply = await self.chat_completion()
        self.messages.append(reply)

        # truncate messages if necessary
        self._truncate_messages()

        return reply["content"]

    @property
    def max_tokens(self) -> int:
        return _MAX_TOKENS[self.model]

    async def chat_completion(self) -> AnyDict:
        try:
            resp = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
            )
        except AuthenticationError as exc:
            raise OpenAIAuthError() from exc
        for choice in resp["choices"]:
            if (reason := choice["finish_reason"]) != "stop":
                print(f"Generation {choice['index']} finished before the end token was reached, {reason=}")

        return [choice["message"] for choice in resp["choices"]][0]

    @classmethod
    def _num_tokens_from_messages(cls, messages: list[AnyDict]) -> int:
        """Returns the approximate number of tokens used by a list of messages."""
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def _truncate_messages(self) -> None:
        num_tokens = self._num_tokens_from_messages(self.messages)
        num_tokens_threshold = self.max_tokens * _MESSAGE_HISTORY_TRUNCATION_THRESHOLD
        if num_tokens > num_tokens_threshold:
            print(
                f"Total tokens {num_tokens} is close to the model's maximum {self.max_tokens}, "
                f"truncating message history"
            )
            # truncate messages until we're below the threshold
            while num_tokens > num_tokens_threshold:
                self.messages = self.messages[1:]
                num_tokens = self._num_tokens_from_messages(self.messages)
