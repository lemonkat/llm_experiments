import dotenv

import util
import client
import persona
import presets

dotenv.load_dotenv()

# load Raiden Shogun's data from the Fandom wiki
raiden_urls = [
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Lore",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Voice-Overs",
    "https://genshin-impact.fandom.com/wiki/Raiden_Shogun/Companion",
]
raiden_text = presets.get_text(raiden_urls)

raiden_desc = presets.get_desc_from_wiki("Raiden", raiden_text, util.GPT4)
print("description loaded")
raiden_quotes = presets.get_quotes_from_wiki("Raiden", raiden_text, util.GPT4)
print("quotes loaded")
raiden_tone = presets.get_tone("Raiden", raiden_quotes, util.GPT4)
presets.save_persona(
    "Raiden",
    "Raiden",
    raiden_desc,
    raiden_quotes,
    f"""Keep responses fairly short - around 1-3 sentences, like in a real conversation.
Respond in a slightly {raiden_tone} tone.""",
    0.4,
)

print("Raiden persona loaded")

# # load Xiangling's data from the Fandom wiki
# xiangling_urls = [
#     "https://genshin-impact.fandom.com/wiki/Xiangling",
#     "https://genshin-impact.fandom.com/wiki/Xiangling/Lore",
#     "https://genshin-impact.fandom.com/wiki/Xiangling/Voice-Overs",
#     "https://genshin-impact.fandom.com/wiki/RXiangling/Companion",
# ]
# xiangling_text = presets.get_text(xiangling_urls)

# xiangling_desc = presets.get_desc_from_wiki("Xiangling", xiangling_text, util.GPT4)
# print("description loaded")
# xiangling_quotes = presets.get_quotes_from_wiki("Xiangling", xiangling_text, util.GPT4)
# print("quotes loaded")
# xiangling_tone = presets.get_tone("Xiangling", xiangling_quotes, util.GPT4)
# presets.save_persona(
#     "Xiangling",
#     "Xiangling",
#     xiangling_desc,
#     xiangling_quotes,
#     f"""Keep responses fairly short - around 1-3 sentences, like in a real conversation.
# Respond in a slightly {xiangling_tone} tone.""",
#     0.4,
# )

# print("Xiangling persona loaded")

raiden = presets.load_persona("Raiden")
# xiangling = presets.load_persona("Xiangling")
user = client.IOClient("User")

print("ready!")


for entry in persona.run_conv([raiden, user]):
    # print(entry)
    pass
