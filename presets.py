import requests
import json

from bs4 import BeautifulSoup as Soup

import util
from client import OpenAIClient
from persona import Persona

DATA_PATH = "data.json"
PRESET_DATA = {}


def get_text(urls: str | list[str]) -> list[str] | list[list[str]]:
    single = isinstance(urls, str)
    urls = [urls] if single else urls
    out = []
    for url in urls:
        html = requests.get(url).text
        soup = Soup(html, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = "\n".join(
            chunk for chunk in chunks if chunk and any(p in chunk for p in ".,!?:;-")
        )
        out.append(text)
    return out[0] if single else out


def save_data():
    global PRESET_DATA
    with open(DATA_PATH, "w") as file:
        json.dump(PRESET_DATA, file, indent=4)


def load_data():
    global PRESET_DATA
    with open(DATA_PATH, "r") as file:
        PRESET_DATA = json.load(file)


def save_persona(
    identifier: str,
    name: str,
    desc: list[str],
    ex: list[str],
    inst: list[str],
    temp: float,
) -> None:
    PRESET_DATA[identifier] = name, desc, ex, inst, temp


def load_persona(identifier: str) -> Persona:
    if identifier == "GPT":
        return OpenAIClient()

    if identifier == "None":
        return None

    if identifier not in PRESET_DATA:
        raise NameError(f"persona {identifier} not found")

    name, desc, ex, inst, temp = PRESET_DATA[identifier]
    return Persona(name, list(set(desc)), inst, list(set(ex)), temp)


def get_desc_from_wiki(
    name: str, texts: str | list[str], model: str = util.GPT3
) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]

    desc = [f"You are {name}."]
    for text in texts:
        prompt = f"""Turn the following webpage text into a set of statements about {name} and what they know.
Pay attention to {name}'s personality and character, not game mechanics or other 4th-wall breaking info.
Place each statement on a seperate line, with no numbering or other prefixes. 
Each statement should be in the second person.
Be sure to mention what is unique about each character.
Be sure to mention what {name} knows about other characters.
Return up to 500 statements.
Example: You are {name}.
Text:
{text}"""
        desc.extend(util.call_LLM(prompt, model).splitlines())
    return desc


#     prompt = f"""This is a list of statements about {name}.
# Remove any redundant statements.
# Place each statement on a seperate line, with no numbering or other prefixes.
# {util.jlines(desc)}"""
#     return list(set(util.call_LLM(prompt, model).splitlines()))


def get_quotes_from_wiki(
    name: str, texts: str | list[str], model: str = util.GPT3
) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]

    quotes = [f"You are {name}."]
    for text in texts:
        prompt = f"""From this webpage, extract all of {name}'s quotes:
{text}
Place each quote on a seperate line, with no numbering or other prefixes.
Be sure to only grab quotes that {name} has said, not quotes about {name}.
if no quotes can be found on this page, return NONE.
"""
        response = util.call_LLM(prompt, model)
        if response.lower() != "none":
            quotes.extend(response.splitlines())

    return quotes


def get_tone(name: str, quotes: list[str], model: str = util.GPT3) -> str:
    prompt = f"""Here are some of {name}'s quotes:
{util.jlines(util.last_n(util.shuffled(quotes), 20))}
In one word, describe the tone / style of these quotes.
Do not explain anything, returning only that single word."""
    return util.call_LLM(prompt, model).strip().lower()


if __name__ == "__main__":
    load_data()
    # from persona import run_conv
    # from client import IOClient

    # for _ in run_conv([IOClient("LemonKat"), load_persona("genshin:Raiden")]):
    #     pass
    data = [  # a couple characters from Genshin Impact
        # ("Beidou", "genshin:Beidou", "Beidou"),
        # ("Venti", "genshin:Venti", "Venti"),
        # ("Zhongli", "genshin:Zhongli", "Zhongli"),
        # ("Raiden", "genshin:Raiden", "Raiden_Shogun"),
        # ("Nahida", "genshin:Nahida", "Nahida"),
        # ("Furina", "genshin:Furina", "Furina"),
        # ("Barbara", "genshin:Barbara", "Barbara"),
        ("Xiangling", "genshin:Xiangling", "Xiangling")
    ]

    for name, identifier, urlname in data:
        print(f"loading {name}")
        urls = [
            f"https://genshin-impact.fandom.com/wiki/{urlname}",
            f"https://genshin-impact.fandom.com/wiki/{urlname}/Lore",
            f"https://genshin-impact.fandom.com/wiki/{urlname}/Voice-Overs",
            f"https://genshin-impact.fandom.com/wiki/{urlname}/Companion",
        ]
        texts = get_text(urls)
        print(f"{name} raw text loaded")
        desc = get_desc_from_wiki(name, texts, util.GPT4)
        print(f"{name} desc loaded")
        quotes = get_quotes_from_wiki(name, texts, util.GPT4)
        print(f"{name} quotes loaded")
        tone = get_tone(name, quotes, util.GPT4)
        save_persona(
            identifier,
            name,
            desc,
            quotes,
            f"Keep responses fairly short - around 1-3 sentences, like in a real conversation. \nRespond in a slightly {tone} tone.",
            0.4,
        )
        save_data()
        print(f"saved {name}")
