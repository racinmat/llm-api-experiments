import asyncclick as click
import pandas as pd
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential
from together import AsyncTogether
from tqdm.asyncio import tqdm_asyncio

from utils import print_results, run_models_parallel, call_model, run_models_file

client = AsyncTogether()


@click.group()
def cli():
    pass


@cli.command("list")
async def list_models():
    models = await client.models.list()
    model_df = pd.DataFrame.from_records([dict(m) for m in models])
    chat_models = model_df[model_df["type"] == "chat"]
    print(tabulate(chat_models[["id", "type", "link"]], headers='keys', tablefmt='pretty'))


@cli.command()
@click.argument("message")
@click.option("--max-tokens", default=200)
async def run(message: str, max_tokens: int):
    results = await run_models(message, max_tokens)
    print_results(results)


async def run_models(message: str, max_tokens: int):
    models = [
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "deepseek-ai/deepseek-llm-67b-chat",
        "microsoft/WizardLM-2-8x22B",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
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
