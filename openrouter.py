import os
import asyncclick as click
import pandas as pd
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from openai import AsyncOpenAI

from utils import print_results, tabulate_df, run_models_parallel, call_model, run_models_file

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
#   model="openai/gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "user",
#       "content": "What is the meaning of life?"
#     }
#   ]
# )
# print(completion.choices[0].message.content)


@click.group()
def cli():
    pass


@cli.command("list")
async def list_models():
    models = list(await client.models.list())[0][1]
    model_df = pd.DataFrame.from_records([dict(m) for m in models])
    tabulate_df(model_df[["id", "name", "description"]], max_col_widths=[50, 70, 130])


@cli.command()
@click.argument("message")
@click.option("--max-tokens", default=200)
async def run(message: str, max_tokens: int):
    results = await run_models(message, max_tokens)
    print_results(results)


async def run_models(message: str, max_tokens: int):
    models = [
        "openai/o1-preview",
        # "openai/gpt-4o-2024-11-20",
        # "anthropic/claude-3.5-sonnet",
        # "google/gemini-2.0-flash-exp:free",
        # "google/gemini-pro-1.5",
        # "x-ai/grok-2-1212",
        # "cohere/command-r-plus-08-2024",
        # "amazon/nova-pro-v1",
        # "qwen/qvq-72b-preview",
        # "01-ai/yi-large",
        # "deepseek/deepseek-r1:free",
        # "deepseek/deepseek-chat",
        # "mistralai/mistral-large-2411",
        # "meta-llama/llama-3.3-70b-instruct",
        # "meta-llama/llama-3.2-90b-vision-instruct:free",
        # "meta-llama/llama-3.1-405b-instruct:free",
        # "microsoft/phi-4",
        # "microsoft/phi-3-medium-128k-instruct:free",
    ]
    tasks = [call_model(client, message, model, max_tokens) for model in models]
    return await run_models_parallel(models, tasks)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--max-tokens", default=200)
async def run_file(file_path: str, max_tokens: int):
    await run_models_file(run_models, file_path, max_tokens)


if __name__ == '__main__':
    cli()
