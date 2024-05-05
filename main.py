import dotenv

import persona

dotenv.load_dotenv()

alice = persona.Persona(
    "Alice",
    ["Your house is on fire."],
    """Keep responses fairly short - around 1-2 sentences, like in a real conversation. 
Also, respond somewhat aggressively - you are trying to convince the others that you are right.""",
    [],
    0.4,
)


bob = persona.Persona(
    "Bob",
    ["Alice's house is NOT on fire."],
    """Keep responses fairly short - around 1-2 sentences, like in a real conversation. 
Also, respond somewhat aggressively - you are trying to convince the others that you are right.""",
    [],
    0.4,
)


for entry in persona.run_conv([alice, bob]):
    print(entry)
