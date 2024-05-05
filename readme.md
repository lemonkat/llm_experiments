### LemonKat's LLM experiments on emulating social interactions and character memories

These are a couple experiments using GPT-3 and GPT-4 to emulate conversations between fictional characters.

Inspired by the paper [_Generative Agents: Interactive Simulacra of Human Behavior_](https://arxiv.org/abs/2304.03442).

Each persona object carries a set of memories of recent events or things they know. When each persona responds, it recalls relevant memories, and sends those along with the conversation history to GPT-4 to generate a response.

To recall relevant memories for a given query, the system assigns each memory a "score", calculated as the sum of:
1. The "importance" of the memory, on a scale from 0.1 to 1.0, given by GPT-3.
2. The "recency" of the memory, which is 1.0 when it is created, and exponentially decays over time.
3. The "relevance" of the memory, which is the cosine similarity between the emebdding vectors of the memory and query.
The recall algorithm picks the top 10 memories with the highest score, and passes those to GPT-4 to produce the response.

For example, take the following conversation between me and a video game character, called Raiden (from Genshin Impact). In-lore, Raiden cannot cook, and this fact is represented in one of her persona's memories. When I ask Raiden whether or not she can cook, the recall algorithm pulls up the relevant memory, and informs GPT-4 to respond that she cannot cook. This all works despite the Raiden persona storing over 50 large memories.

>```
>LemonKat: can you cook?  
>Raiden: I do not possess culinary skills. My focus lies elsewhere, in the governance of Inazuma and the pursuit of eternity. However, I am intrigued by the art of dessert making and its required patience.
>```

#### To run:
1. create a `.env` file with `OPENAI_API_KEY` set to your OpenAI API key.
2. run `python3 main.py`