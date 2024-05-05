import random
from typing import Callable

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import dotenv

from openai import OpenAI

GPT3 = "gpt-3.5-turbo"
GPT4 = "gpt-4-0125-preview"

USER = 0
AI = 1
SYSTEM = 2

dotenv.load_dotenv()

BACKEND = OpenAI()

def call_LLM(query: str | list[tuple[int, str]] | list[str] | list[list[tuple[int, str]]], model: str = GPT4, temp: float = 0.0, single: bool = True) -> str | list[str]:
    if single:
        if isinstance(query, str):
            query = [(USER, query)]
        return _call_LLM(query, model, temp)
    
    if isinstance(query[0], str):
        query = [[(USER, q)] for q in query]
    
    with ThreadPoolExecutor(max_workers=100) as exc:
        result = [None] * len(query)
        future_to_idx = {exc.submit(_call_LLM, q, model, temp): i for i, q in enumerate(query)}
        for future in as_completed(future_to_idx):
            result[future_to_idx[future]] = future.result()
        return result

# does final processing and actual call - seperate from threading
def _call_LLM(query: list[tuple[int, str]], model: str, temp: float) -> str:
    return BACKEND.chat.completions.create(
        messages=[{"role": ["user", "assistant", "system"][t], "content": c} for t, c in query], 
        model=model, 
        temperature=temp
    ).choices[0].message.content


def shuffled(lst):
    result = list(lst)
    random.shuffle(result)
    return result

# powerful utility function
def last_n(lst: list, n: int, key: Callable[[object], float] | str | None = None, reverse: bool = False) -> list:
    """
    returns the last n items of a list.
    if reverse is True, then grab the first ones.
    key will sort be used to sort the items beforehand.
    if key == None, no sorting will be applied.
    if key == "random", then will return n random items. 
    """
    if n >= len(lst):
        return lst
    if key is None:
        return lst[:n] if reverse else lst[-n:]
    if key == "random":
        return random.sample(lst, n)
    return sorted(lst, key=key, reverse=reverse)[-n:]

# bc you can't have \n in f-strings for some reason
def jlines(lines: list[str]) -> str:
    return "\n".join(map(str, lines))

def get_embedding(text: str | list[str], norm: bool = True) -> list[float] | list[list[float]]:
    """
    returns an embedding vector of the text. 
    This can be used to compute the similarity of the meanings of 2 strings.
    """
    single, text = (True, [text]) if isinstance(text, str) else (False, text)
    text = [t.replace("\n", " ") if t else "this is blank" for t in text]
    # response = BACKEND.embeddings.create(input=text, model="text-embedding-3-small")
    response = BACKEND.embeddings.create(input=text, model="text-embedding-3-large")
    result = np.array([emb.embedding for emb in response.data], dtype=np.float32)
    if norm:
        result = result / np.linalg.norm(result, axis=1)[:, None]
    return result[0] if single else result

def cos_sim(a: list[float], b: list[float], norm: bool = False) -> float:
    """
    returns the cosine similarity between 2 vectors, 
    which is an approximate measure of how close they are.
    """
    if norm:
        return np.dot(a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.dot(a, b)
    
# def get_keywords(text: str | list[str], k: int = 5, model: str = GPT3) -> set[str] | list[set[str]]:
#     # kwds = [token.strip().lower() for token in desc.split() if token[0].isupper()]
#     # return random.sample(kwds, k)
#     single, text = (True, [text]) if isinstance(text, str) else (False, text)
#     prompts = [
#         f"""Return up to {k} space-seperated keywords for the content of following quotation:
# {t}
# Be sure to mention names and other unique features."""
#         for t in text
#     ]
#     result = [set(response.strip().lower().split()) for response in call_LLM(prompts, model)]
#     return result[0] if single else result

if __name__ == "__main__":
    print(call_LLM("what is Obama's last name?"))