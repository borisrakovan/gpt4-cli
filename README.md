# GPT-4 CLI

Simple command line interface for chatting with the newest GPT-4 language model using the official OpenAI API

To get started, clone the repository and install the dependencies using pip or poetry.
    
Then, create a .env file in the root directory of the project and add your OpenAI API key to it:

```
OPENAI_API_KEY=<your_api_key>
```

Alternatively, you can also set the `OPENAI_API_KEY` environment variable to your API key.

For multiline prompts, you can start the prompt with `>` and follow it with a filename on your local disk. The prompt
will then be loaded from the provided file instead of the command line.