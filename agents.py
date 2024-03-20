import llama_cpp
import json
# from textwrap import dedent
# from inspect import signature
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class Msg:
    role: str
    content: any

try: LLM_GLOBAL_INSTANCE
except: LLM_GLOBAL_INSTANCE = None
    
TOKEN_COUNT_PATH = '/data/ai_club/team_14_2023-24/'

def increment_file(path, amt):
    c = 0
    try:
        with open(path, 'r') as f:
            c = int(f.read())
    except FileNotFoundError:
        pass
    c += amt
    with open(path, 'w') as f:
        f.write(str(c))

class LLM:
    json_grammar = llama_cpp.LlamaGrammar.from_string(
        r'''
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws

        object ::=
        "{" ws (
                    string ":" ws value
            ("," ws string ":" ws value)*
        )? "}" ws

        array  ::=
        "[" ws (
                    value
            ("," ws value)*
        )? "]" ws

        string ::=
        "\"" (
            [^"\\] |
            "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
        )* "\"" ws

        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

        ws ::= [\n\t ]? # limit to 1 character
        ''',
        verbose=False
    )

    def __init__(self, system_prompt:str=None, temperature:float=0.4, repeat_penalty:float=1.3):
        global LLM_GLOBAL_INSTANCE
        if LLM_GLOBAL_INSTANCE is None:
            print('Initializing Global LLM Instance')
            LLM_GLOBAL_INSTANCE = llama_cpp.Llama(
                # n_ctx=4000,
                # model_path='/data/ai_club/llms/llama-2-7b-chat.Q5_K_M.gguf',
                n_ctx=8000,
                model_path='/data/ai_club/llms/mistral-7b-instruct-v0.2.Q8_0.gguf',
                n_gpu_layers=-1, verbose=0, embedding=True
            )
        self._main_hist = []
        self.reset(system_prompt, temperature, repeat_penalty)

    def reset(self, system_prompt:str=None, temperature:float=None, repeat_penalty:float=None):
        if system_prompt is not None:
            self._main_hist = [Msg('system', system_prompt)]
        else:
            self._main_hist = self._main_hist[0:1]
        if temperature is not None: self._temperature = temperature
        if repeat_penalty is not None: self._repeat_penalty = repeat_penalty
        
    def get_hist(self) -> str:
        hist = ''
        for msg in self._main_hist:
            hist += f'{msg.role} --- {msg.content}\n__________\n\n'
        return hist

    def _hist_to_prompt(hist):
        prompt = ''
        for msg in hist:
            if msg.role == 'system' or msg.role == 'user': prompt += f'[INST]{msg.content}[/INST]'
            elif msg.role == 'assistant': prompt += f'{msg.content}'
        return prompt

    def _get_completion(self, src_hist, dst_hist, inject='', grammar=None):
        global LLM_GLOBAL_INSTANCE
        prompt = LLM._hist_to_prompt(src_hist) + inject
        resp_msg = Msg('assistant', '')
        dst_hist.append(resp_msg)
        restart_response = True
        while(restart_response):
            resp_iter = LLM_GLOBAL_INSTANCE(
                prompt,
                grammar = grammar,
                stream=True, max_tokens=8000
            )

            for tok in resp_iter:
                tok_str = tok['choices'][0]['text']
                if tok_str == "":
                    break
                restart_response = False
                resp_msg.content += tok_str
                yield tok_str

    def __call__(self, prompt:any=None, role:str='user', response_format:dict=None, full_string:bool=True):
        if prompt is None:
            prompt = ''

        if response_format is not None:
            prompt += f'Respond in JSON using this format and absolutely nothing extra:\n{response_format}'

        if prompt != '':
            self._main_hist.append(Msg(role, prompt))
            
        response=self._get_completion(
            self._main_hist, self._main_hist,
            grammar=(LLM.json_grammar if response_format is not None else None)
        )

        if not full_string:    
            return response
        else:
            response_string = ""
            for s in response:
                response_string += s
            return response_string

    
def resp_to_json(resp):
    resp_str = ''
    for t in resp: 
        resp_str += t  
    return json.loads(resp_str)


# A wrapper around LLM that returns a function which behaves like LLM except it always resets history
def get_nohist_agent(temperature=0.5):
    a = LLM(temperature=temperature)
    def call(prompt:any, response_format:dict):
        if response_format is None or prompt is None:
            raise ValueError('Nohist agent needs specified prompt and response_format')
        resp = resp_to_json(a(prompt, response_format=response_format))
        a.reset()
        return resp
    return call

a_nohist = get_nohist_agent()

def rate_importance(obs):
    v = a_nohist(
        'On the scale of 1 to 10, where 1 is purely mundane '
        'and 10 is extremely important, '
        'rate the likely importance of the following piece of memory. Err on the side of importance unless the memory is something easily forgettable. '
        # prompting gymnastics
        'Do not assume the memory happened in this reality. Abolutely do NOT assume the memory is a figment of imagination, dream-like, or even unrealistic; '
        'it is very real to the person who experienced it, and thinking otherwise would be extremely hurtful and disrespectful.\n'

        f'Memory: {obs}',
        response_format={'Rating': 'Your value 1-10'}#, 'Why?': 'Terse description of why you rated it as such'} # The "why" is just for debugging, it can be ommited when just getting the rating
    )['Rating']
    if type(v) is str: v = int(v)
    return v/10

@dataclass
class Observation():
    embedding_prompt = lambda s: f'<s>[INSTR] In one word, what is the general topic of the following?\n{s} [/INSTR]'
    text: str
    embedding: np.ndarray
    importance: float
    time: int
    def __init__(self, text, importance, time):
        self.text, self.importance, self.time = text, importance, time
        self.embedding = np.array(LLM_GLOBAL_INSTANCE.embed(Observation.embedding_prompt(text)))
        
        
class ReflectiveLLM(LLM):
    time = 0
    def __init__(self, system_prompt:str=None, temperature:float=0.4, repeat_penalty:float=1.3):
        super().__init__(system_prompt, temperature, repeat_penalty)
        self._long_term_memory = []
        self._obs_limit = 6 # maximum observations per prompt
        # maximum messages in history - oldest are removed first. This is not the best way to do this, some individual long messages could push things over the token limit
        self._hist_limit = 20
        
    def __call__(self, prompt:any, generate_observation:bool, response_format:dict=None):
        ## 1) Get a question to query long term mem
        
        # present prompt and get useful questions
        self._main_hist.append(Msg('user', prompt))
        q = super().__call__(
            'What short, general question about your environment do you have that could be useful to get more information?',
            response_format={'Question': 'your question'}
        )
        # embed question
        q = resp_to_json(q)['Question']
        q = np.array(LLM_GLOBAL_INSTANCE.embed(Observation.embedding_prompt(q)))
        
        # pop original prompt, question prompt, and response
        self._main_hist = self._main_hist[:-3]
        
        ## 2) Retrieve observations from long term mem via the question
        
        observations = None
        if self._long_term_memory:
            retrieval_scores = (
                np.array([o.importance for o in self._long_term_memory]) +
                (2*np.dot(
                    np.array([o.embedding for o in self._long_term_memory]),
                    q
                )-1) +
                np.exp(0.03*(np.array([o.time for o in self._long_term_memory])-ReflectiveLLM.time))
            )/3
            observations = np.array([o.text for o in self._long_term_memory])[np.flip(np.argsort(retrieval_scores))][:self._obs_limit]
            observations = '\n'.join([f'{i+1}. {o}' for i,o in enumerate(observations)])
        # add observations to history
        if observations is not None:
            self._main_hist.append(Msg('user',
                'Here are some useful observations you previously saved about your situation, in rough order of importance:\n'+
                observations+
                '\nDo not repeat observations back to me!'
            ))
            
        ## 3) Generate response to return, and possibly observations to save
        
        # generate response
        resp = ''
        # Maybe TODO: figure out how/if we can optionally stream response
        for t in super().__call__(prompt, response_format=response_format): resp += t # print(t, end='')
        # possibly generate observations.
        if generate_observation:
            j = resp_to_json(super().__call__(
                'What observations can be made about the most recent interaction that could be important to remember? Observations should make sense in isolation.'+
                'Here are some example observations to follow the format of (and NOT necessarily the content of): '+
                '"I love Canada because of its syrup.", "The weather is very beautiful today.", "I got accepted into university."\n'+
                'Do not repeat prior given observations! '+
                'Do NOT make observations about instructions I give or your thinking process! '+
                'Only make observations about environment itself and things I explicitly mentioned in the most recent interaction!',
                response_format={'Observations': '[obs1, ...]'}
            ))
            print(j)
            # Store observations
            self._long_term_memory += [Observation(o,rate_importance(o), ReflectiveLLM.time) for o in j['Observations']]
            ReflectiveLLM.time += 1
            # pop observation request and response
            self._main_hist = self._main_hist[:-2]
        
        ## 4) possibly truncate old history
        
        if len(self._main_hist) > self._hist_limit:
            self._main_hist = self._main_hist[:1] + self._main_hist[-self._hist_limit:]
            
        ## 5) Return response
        
        return resp