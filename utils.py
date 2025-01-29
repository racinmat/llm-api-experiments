from typing import Coroutine
from together import AsyncTogether
from openai import AsyncOpenAI
import pandas as pd
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

# try:  # this is to handle the case when the terminal is too narrow
#     columns, rows = os.get_terminal_size()
# except OSError:
#     columns, rows = 220, 40
columns = 250  # I don't want to use whole width, I want to have it readable on GitHub too...


def print_results(results: list[dict]):
    result_df = pd.DataFrame(results)
    first_col = 50
    tabulate_df(result_df, ["model", "answer"], [first_col, columns - first_col - 2])


def tabulate_df(df: pd.DataFrame, headers: list[str] = None, max_col_widths: list[int] = None):
    if headers is None:
        headers = df.columns
    if max_col_widths is None:
        max_col_widths = [columns // len(df.columns)] * len(df.columns)
    print(tabulate(df, headers=headers, tablefmt='pretty', showindex=False, maxcolwidths=max_col_widths))


async def run_models_parallel(models: list[str], tasks: list[Coroutine]):
    answers = await tqdm_asyncio.gather(*tasks, desc="Querying models")
    results = []
    for result, model in zip(answers, models):
        if result.choices:
            response = result.choices[0].message.content or ""
            # provider = result.provider
        else:
            response = "No response"
        results.append({"model": model, "answer": response})
    return results


@retry(wait=wait_exponential(min=1, multiplier=1, exp_base=1.2, max=60), stop=stop_after_attempt(30),
       # after=lambda x: print(f'Retrying after {x}')
       )
async def call_model(client: AsyncTogether | AsyncOpenAI, message: str, model: str, max_tokens: int):
    # print('Calling model:', model + ' with message:', message)
    res = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=False,
        max_tokens=max_tokens,
    )
    # print('Finished model:', model + ' with message:', message)
    return res


async def run_models_file(run_models, file_path, max_tokens):
    with open(file_path, 'r', encoding='utf-8') as file:
        # stripping empty rows
        questions = [question.strip() for question in file.readlines() if question.strip()]
    print(f"Found {len(questions)} questions in the file")
    # no need to parallelize this, the inner is parallel, and I want to see the results as they appear
    for question in questions:
        print(f"Question: {question}")

        result = await run_models(question.strip(), max_tokens)
        print_results(result)
