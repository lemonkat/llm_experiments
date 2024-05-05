import dotenv

import util
import client
import persona
import presets

dotenv.load_dotenv()

# load Raiden Shogun's data from the Fandom wiki
urls = [
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Lore",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Voice-Overs",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Companion",
]

desc = presets.get_desc_from_wiki("Raiden", urls, util.GPT4)
print("description loaded")
quotes = presets.get_quotes_from_wiki("Raiden", urls, util.GPT4)
print("quotes loaded")
tone = presets.get_tone("Raiden", quotes, util.GPT4)
presets.save_persona(
    "Raiden",
    "Raiden",
    desc,
    quotes,
    f"""Keep responses fairly short - around 1-3 sentences, like in a real conversation.
Respond in a slightly {tone} tone.""",
    0.4,
)
print("ready!")

user = client.IOClient("User") # follows interface of a Persona, but lets the user interact

for entry in persona.run_conv([user, presets.load_persona("Raiden")]):
    # print(entry)
    pass
