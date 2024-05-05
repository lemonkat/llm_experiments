from abc import ABC, abstractmethod
import sys

import util

class Client(ABC):

    # name is just so the client object can easily be referred to in text
    # such as for use in conversations
    def __init__(self, name: str = None):
        self.name = name
        self._history = []
        self._ready = None

    @abstractmethod
    def _read(self) -> str:
        """For subclasses to implement. Gets a response from the AI."""
        return ""

    # can simply do nothing if the AI does everything on-read
    def _write(self, data: str) -> None:
        """For subclasses to implement. Gives information to the AI."""
        return

    def read(self) -> str:    
        """Gets a response from the AI."""
        response = self._read()
        self._history.append((1, response))
        self._ready = None
        return response
    
    def write(self, query: str, msg_type: int = util.USER) -> str:
        """Gives information to the AI."""
        if not isinstance(query, str):
            raise TypeError(f"Query {query} is not of type str")
        
        self._history.append((msg_type, query))
        self._write(query)
        self._ready = None

    @property
    def history(self) -> list[tuple[int, str]]:
        """a list of tuples of [type, content] storing this client's history of calls. """
        return self._history
    
    def call(self, query: str) -> str:
        """
        Sends <query> to the LLM and returns what the LLM responds.
        same as using write() followed by read(). """
        self.write(query)
        return self.read()
    
    def single_call(self, query: str) -> str:
        """
        Sends <query> to the LLM and returns what the LLM responds. 
        Doesn't use the history - just a single query and response.
        Same as calling clear() followed by call(), or clear() then write() then read().
        """
        self.clear()
        return self.call(query)
    
    def clear(self):
        """clears the client's history."""
        self._history.clear()
        self._ready = None

    @property
    def is_ready(self) -> bool:
        """whether this client "wants" to say something right now"""
        if self._ready is None:
            self._ready = self._is_ready()
        return self._ready

    def _is_ready(self) -> bool:
        """returns whether this client "wants" to say something right now"""
        return True

class IOClient(Client):
    """
    A basic client that simply refers you to the provided IO streams.
    By default, it will use stdin / stdout for communicating with a user via the terminal.
    """
    def __init__(self, name: str = "User", inp = sys.stdin, out = sys.stdout):
        super().__init__(name)
        self.inp, self.out = inp, out

    def _read(self) -> str:
        # return self.inp.readline()
        return input("You: ")

    def _write(self, data: str) -> None:
        # self.out.write(data + "\n")
        print(data)

    def _is_ready(self) -> bool:
        # self.out.write("want to say something? [y/n]: ")
        # self.out.flush()
        # return self.inp.readline().strip().lower().startswith("y")
        return input("want to say something? [y/n]: ").strip().lower().startswith("y")

class OpenAIClient(Client):
    def __init__(
            self, 
            name: str = "GPT-4",
            system_msg: str = None,
            temp: float = 0,
            model: str = "gpt-4-0125-preview",
        ):
        super().__init__(name)

        assert 0 <= temp <= 1, "Temperature must be between 0 and 1"

        self._system_msg = system_msg
        self._temp = temp
        self._model = model
        self.clear()

    def _read(self) -> str:
        # for t, d in self.history:
        #     messages.append({"role": ["user", "assistant", "system"][t], "content": d})

        # try:
        #     comp = self._client.chat.completions.create(
        #         messages=messages, 
        #         model=self._model,
        #         temperature=self._temp,
        #     )

        # except openai.APIConnectionError:
        #     raise ConnectionError("Could not connect to API.")
        
        # except openai.AuthenticationError:
        #     raise KeyError("Invalid API Key.")
        
        # except openai.RateLimitError:
        #     raise CapitalismException()
        
        # return comp.choices[0].message.content
        return util.call_LLM(self.history, temp=self._temp)
    
    def clear(self):
        super().clear()
        if self._system_msg is not None:
            self.write(self._system_msg, 2)

if __name__ == "__main__":
    cl = OpenAIClient()
    while True:
        cl.write(input("User: "))
        print(f"AI: {cl.read()}")