import os
import asyncclick as click
import pandas as pd
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from openai import AsyncOpenAI

from utils import print_results

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="openai/gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)


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
    results = await run_models(client, message, max_tokens)
    print_results(results)


async def run_models(client: AsyncTogether, message: str, max_tokens: int):
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
    answers = await tqdm_asyncio.gather(*tasks, desc="Querying models")
    results = []
    for result, model in zip(answers, models):
        if result.choices:
            response = result.choices[0].message.content or ""
        else:
            response = "No response"
        results.append({"model": model, "answer": response})
    return results


@retry(wait=wait_exponential(min=1, multiplier=1, exp_base=1.2, max=60), stop=stop_after_attempt(30),
       # after=lambda x: print(f'Retrying after {x}')
       )
async def call_model(client: AsyncTogether, message: str, model: str, max_tokens: int):
    # print('Calling model:', model + ' with message:', message)
    res = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=False,
        max_tokens=max_tokens,
    )
    # print('Finished model:', model + ' with message:', message)
    return res


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--max-tokens", default=200)
async def run_file(file_path: str, max_tokens: int):
    with open(file_path, 'r', encoding='utf-8') as file:
        # stripping empty rows
        questions = [question.strip() for question in file.readlines() if question.strip()]
    print(f"Found {len(questions)} questions in the file")

    client = AsyncTogether()

    # no need to parallelize this, the inner is parallel, and I want to see the results as they appear
    for question in questions:
        print(f"Question: {question}")

        result = await run_models(client, question.strip(), max_tokens)
        print_results(result)


if __name__ == '__main__':
    cli()
