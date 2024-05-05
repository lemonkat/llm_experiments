from typing import Iterable
import random

import util
from client import Client

def get_importance(mem_type: int, desc: str) -> float:
    """
    Returns the perceived importance of a memory.
    """
    if mem_type == Memory.IDENTITY:
        return 1.0
    
    if mem_type == Memory.QUERY:
        return 0.0
    
    response = util.call_LLM(f"""On the scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the
following piece of memory. Only return the number, do not explain anything.
Memory: {desc}
Rating: <fill in>""", util.GPT3)
    try:
        return int(response) / 10
    except ValueError:
        return random.random()

class Memory:
    """
    This class represents a memory, 
    with an embedding vector, creation and access timestamps, an importance score, 
    and references to other memories.
    """
    IDENTITY = 0
    OBSERVATION = 1
    REFLECTION = 2
    ACTION = 3
    PLAN = 4
    SUMMARY = 5
    QUERY = -1

    MEM_TYPES = {
        IDENTITY: "Identity",
        OBSERVATION: "Observation",
        REFLECTION: "Reflection",
        ACTION: "Action",
        PLAN: "Plan",
        SUMMARY: "Summary",
        QUERY: "Query"
    }

    # QUERY memories aren't true memories - they're used to look for other ones

    def __init__(self, mem_type: int, desc: str, time: int, ref: list["Memory"] = []):
        self.type = mem_type
        self.desc = desc
        self.ref = ref

        self.crt = self.acc = time

        for mem in ref:
            mem.acc = time
        
        self.refcount = 0

        self.imp = get_importance(self.type, self)

        self._emb = None
        self._has_emb = False

    def get_emb(self) -> list[float]:
        if not self._has_emb:
            if self.type == self.QUERY and len(self.ref):
                self._emb = sum(r.get_emb() for r in self.ref)

            else:
                self._emb = util.get_embedding(self.desc)

            self._has_emb = True
        return self._emb
    
    # so embeddings can be batch-processed by an external system
    def set_emb(self, emb: list[float]) -> None:
        self._emb = emb
        self._has_emb = True

    def __str__(self) -> str:
        return f"{self.MEM_TYPES[self.type]} at t={self.crt}: {self.desc}"
    
    def sys_str(self) -> str:
        return f"""{self}
type={self.type}
importance={self.imp}
created at t={self.crt}, last accessed at t={self.acc}
Refrences:
{util.jlines(self.ref)}"""



class Persona(Client):

    """
    This client emulates a person with memories.
    They can plan out and reflect on their actions, 
    as well as summarize older memories. (TODO: do these things)
    """

    def __init__(self, name: str, id: list[str], instructions: str, examples: list[str], temp: float):
        super().__init__(name)

        self.temp = temp

        self.identity = id
        self.inst = instructions

        self.identity_mem = None
        self.examples = examples
        
        self.mem = []

        self.clear()
        print(f"ready: {self.name}")

    def add_mem(self, mem_type: int, data: str, ref: list[Memory] = []) -> Memory:
        """Add the following memory to the memory stream."""
        mem = Memory(mem_type, data, self.time, list(ref))
        self.mem.append(mem)
        return mem
        
    def clear(self) -> None:
        super().clear()
        self.time = 0

        if self.identity_mem is None:
            self.mem = []

            for desc, emb in zip(self.identity, util.get_embedding(self.identity)):
                self.add_mem(Memory.IDENTITY, desc).set_emb(emb)

            self.identity_mem = self.mem[:]
    
        else:
            self.mem = self.identity_mem[:]
        
    def _read(self) -> str:
        recent = util.last_n([m for m in self.mem if m.type != Memory.IDENTITY], 10, key=lambda m: m.crt)
        # print(f"RECENT:\n{util.jlines(recent)}")
        recalled = sorted(self.recall(recent, 10) + recent, key=lambda mem: mem.crt)
        ex_str = f"""Carefully mimic the style and tone of these examples:
{util.jlines(util.last_n(self.examples, 5, key="random"))}""" if self.examples else ""
        
        for m in recalled:
            m.acc = self.time

        prompt = f"""Your recent memories:
{util.jlines(recalled)}
What would you do or say in the current situation?
If you would say something, only return what you say, without enclosing quotation marks.
Act like the character described, NOT like an assistant.
{ex_str}
{self.inst}
{self.name}: """
        response = util.call_LLM(
            [
                (util.SYSTEM, f"You are {self.name}."),
                (util.USER, prompt)
            ],
            model=util.GPT4,
            temp=self.temp,
        )  
        # response = self.backend.single_call(prompt)
        self.add_mem(Memory.ACTION, response)
        self.time += 1

        return response

    def _write(self, data: str) -> None:
        self.add_mem(Memory.OBSERVATION, data)
        self.time += 1

    def _is_ready(self) -> bool:
        recent = util.last_n([m for m in self.mem if m.type != Memory.IDENTITY], 10, key=lambda m: m.crt)
        prompt = f"""You are {self.name}.
Here are your recent memories:
{util.jlines(recent)}
Would you want to say something in the current conversation?
Return YES or NO. Do not give an explanation."""
        return util.call_LLM(prompt, util.GPT3).strip().lower().startswith("y")

    # more advanced actions, not currently in use
    def reflect(self) -> None:
        """Reflect on recent events, generating new memories."""
        recent = util.last_n(self.m, 20, key=lambda m: m.acc)
        prompt = f"""Here are {self.name}'s recent memories:
{util.jlines(recent)}
“Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?
Place each question on a seperate line."""

        queries = [Memory(Memory.QUERY, desc, self.time, recent) for desc in util.call_LLM(prompt).splitlines()]
        refs = self.recall(queries, 10)
        
        mem_str = util.jlines(f"{i + 1}. {mem}" for i, mem in enumerate(refs))
        prompt = f"""Statements about you, {self.name}:
{mem_str}
What high-level insights can you infer from the above statements? 
Return up to 5 statements, with each statement on a seperate line. 
Do not number your statements.
""" 
        reflections = []
        for stmt in util.call_LLM(prompt).splitlines():
            if stmt.strip() != "":
                reflections.append(self.add_mem(Memory.REFLECTION, stmt, refs))
                print(f"{self.name} - REFLECTION: {stmt}")
        
        prompt = f"""Statements about you, {self.name}:
{util.jlines(m for m in self.mem if m.type == Memory.IDENTITY)}
{util.jlines(reflections)}
Your recent experiences:
{util.jlines(recent)}
Given these, what are some actions you should take in the near future?
(e.g. things you would say, questions you would ask, actions you would take)
Return up to 3 actions, with each actions on a seperate line. 
Do not number your actions.
Each action should be of the form: {self.name} should <action>.
"""
        refs = recent + reflections
        for action in util.call_LLM(prompt).splitlines():
            if action.strip() != "":
                self.add_mem(Memory.PLAN, action, refs)
                print(f"{self.name} - PLAN: {action}")
    
        self.time += 1

    def recall_score(self, query: Memory, key: Memory) -> float:
        """Get the recall score between the given query memory and the given key memory."""
        relevance = util.cos_sim(query.get_emb(), key.get_emb()) # cosine-based similarity
        recency = 0.995 ** (self.time - key.acc) # brought up / used recently?
        importance = key.imp # important?
        return 5 * relevance + 2 * recency + 3 * importance

    def recall(self, query: Memory | list[Memory], k: int) -> list[Memory]:
        """Find a list of memories that are likely related to the given memory."""
        used = query if isinstance(query, list) else [query]
        query = Memory(Memory.QUERY, "[QUERY]", 0, query) if isinstance(query, list) else query
        result = util.last_n([m for m in self.mem if m not in used], k, key=lambda mem: self.recall_score(query, mem))
        # print("RECALLED:\n" + util.jlines(result))
        return result
    
def run_conv(personas: list[Persona]) -> Iterable[str]:
    persona = None
    for p in personas:
        others = [q.name for q in personas if q != p]
        if len(others) == 1:
            others_str = others[0]
        else:
            others_str = ", ".join(others[:-1]) + ", and " + others[-1]

        p.write(f"You are in a conversation with {others_str}.")

    if len(personas) == 2:
        s = f"{personas[0].name}: {personas[0].read()}"
        yield s
        while True:
            s = f"{personas[1].name}: {personas[1].call(s)}"
            yield s
            s = f"{personas[0].name}: {personas[0].call(s)}"
            yield s
            
            

    while True:
        choices = [p for p in personas if p != persona and p.is_ready]
        choices = choices if choices else [p for p in personas if p != persona]

        persona = random.choice(choices)
        response = persona.read()
        s = f"{persona.name}: {response}"

        for p in personas:
            if p != persona:
                p.write(s)
        
        yield s
