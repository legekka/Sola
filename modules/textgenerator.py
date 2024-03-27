import llama_cpp_cuda as llama_cpp
import random
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
import aiohttp

import re
import json

emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)

def remove_emojis(text):
    return emoji_pattern.sub(r'', text)

class TextGenerator:
    def __init__(self, config={"model_path": "E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf", "format_path": "formats/ChatML.json"}):
        self.model = llama_cpp.Llama(model_path=config["model_path"],
                        n_ctx=4096,
                        n_batch=512,
                        n_threads=6,
                        n_gpu_layers=128,
                        offload_kqv=True,
                        use_mmap=True,
                        mul_mat_q=True,
                        #verbose=False
                        )
        with open(config["format_path"], "r") as f:
            self.template = json.load(f)
        with open("prompts/system_rephrase.txt", "r") as f:
            self.system_rephrase_prompts = f.read().splitlines()
        with open("prompts/system_chat.txt", "r") as f:
            self.system_chat_prompts = f.read().splitlines()
        
        if "biases" in config:
            self.biases = config["biases"]
        elif config["model_path"] == "E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf":
            self.biases = {
                "315": -10,   # " I"
                "630": -0.5,  # " she"
                "478": -0.5,  # " we"
                "816": -0.5,  # " We"
                "985": -0.5,  # " She"
                "3489": -0.5, # " Our"
            }
        else:
            self.biases = None
        self.chat_init = ["Initializing ship systems, preparing to launch.", "Sola is getting all the spaceship systems ready, and preparing for a big, exciting launch!"]
        self.chat_history = self.chat_init.copy() 

        # self.temperature = 0.7
        # self.top_p = 0.9
        # self.top_k = 20
        # self.min_p = 0
        # self.frequency_penalty = 0
        # self.presence_penalty = 0
        # self.repeat_penalty = 1.15
        # self.typical_p = 1

        self.temperature = 1
        self.top_p = 1
        self.top_k = 0
        self.min_p = 0.1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.repeat_penalty = 1
        self.typical_p = 1
        self.mirostat_mode = 0
        self.mirostat_tau = 5
        self.mirostat_eta = 0.1

    def format_prompt(self, prompt, history=None):
        prompt = prompt.strip()
        if self.template["type"] == "chat":
            # we start with the system prompt
            text = self.template["system_prefix"] + random.choice(self.system_prompts) + self.template["end_token"]

            # if history provided, we add each pair of history to the prompt
            if history is not None:
                for i in range(0, len(history), 2):
                    text += self.template["user_prefix"] + history[i] + self.template["end_token"]
                    text += self.template["assistant_prefix"] + history[i+1] + self.template["end_token"]
        
            # and then we add the current user prompt and the assistant prefix for the next response
            text += self.template["user_prefix"] + prompt + self.template["end_token"]
            text += self.template["assistant_prefix"]
        
        elif self.template["type"] == "instruct":
            # we start with the system prompt
            text = self.template["bos_token"] + self.template["instruct_prefix"] + "SYSTEM: " + random.choice(self.system_prompts) + " | "

            # if history provided, we add each pair of history to the prompt
            if history is not None:
                # the first is different, because of the system prompt
                text += self.template["user_prefix"] + history[0] + self.template["instruct_suffix"]
                text += self.template["assistant_prefix"] + history[1] + self.template["end_token"]
                for i in range(2, len(history), 2):
                    text += self.template["instruct_prefix"] + self.template["user_prefix"] + history[i] + self.template["instruct_suffix"]
                    text += self.template["assistant_prefix"] + history[i+1] + self.template["end_token"]

                # and then we add the current user prompt and the assistant prefix for the next response
                text += self.template["instruct_prefix"] + self.template["user_prefix"] + prompt + self.template["instruct_suffix"]
                text += self.template["assistant_prefix"]
            else:
                text += self.template["user_prefix"] + prompt + self.template["instruct_suffix"]
                text += self.template["assistant_prefix"]

        return text

    def clean_text(self, text):
        # removes any whitespace at the start and end of the text
        text = text.strip()

        # there are words with - in them, like "well-being", we want to separate them into "well being" (removing the "-"). But only if there's no spaces between the - and the words
        text = re.sub(r"(\w)-(\w)", r"\1 \2", text)
    
        # check if there are any decimal numbers (like 0.22), and replace the "." to " point "
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)

        # also if there are numbers with a comma (like 12,000), remove the comma to be 12000
        text = re.sub(r"(\d+),(\d+)", r"\1\2", text)

        return text

    def split_text(self, text):
        # splits text into multiple sentences based on punctuation ".", "!", "?", ";", ":", and newline "\n"
        # it's important to check the next character after the punctuation, because we don't want to split on for example numbers

        text = self.clean_text(text)

        sentences = []
        current_sentence = ""

        for i in range(len(text)):
            current_sentence += text[i]
            if text[i] in [".", "!", "?", ";", ":", "\n"] and i+1 < len(text) and (text[i+1] == " " or text[i+1] == "\n"):
                sentences.append(current_sentence)
                current_sentence = ""
        if current_sentence != "":
            sentences.append(current_sentence)

        # A section should be less than 150 characters, one exception:
        # - If a section would be shorter than 5 words, it can be combined with the next section, or the previous section if it's the last section

        sections = []
        for i in range(len(sentences)):
            if len(sentences[i]) < 150:
                if i+1 < len(sentences):
                    if len(sentences[i] + sentences[i+1]) < 150:
                        sentences[i+1] = sentences[i] + sentences[i+1]
                    else:
                        if len(sentences[i].split(" ")) < 7:
                            sentences[i+1] = sentences[i] + sentences[i+1]
                        else:
                            sections.append(sentences[i])
                elif len(sentences[i].split(" ")) < 7:
                    sections[-1] += sentences[i]
                else:
                    sections.append(sentences[i])
            else:
                sections.append(sentences[i])

        # check if the last section is less than 5 words, if so, combine it with the previous section
        if len(sections) > 1:
            if len(sections[-1].split(" ")) < 5:
                print("Combining last two sections")
                sections[-2] += sections[-1]
                sections.pop(-1)

        for i in range(len(sections)):
            print("Section length: " + str(len(sections[i])) + " characters, " + str(len(sections[i].split(" "))) + " words")

        # strip whitespace from the sentences
        for i in range(len(sections)):
            sections[i] = sections[i].strip()
        return sections

    def rephrase(self, text, max_tokens=150):
        self.system_prompts = self.system_rephrase_prompts
        prompt = self.format_prompt(text)
        start = time.time()
        response = self.model(prompt=prompt,
                              stop=self.template["end_token"],
                              echo=False,
                              logit_bias=self.biases if self.biases is not None else None,
                              max_tokens=max_tokens,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              min_p=self.min_p,
                              top_k=self.top_k,
                              frequency_penalty=self.frequency_penalty,
                              presence_penalty=self.presence_penalty,
                              repeat_penalty=self.repeat_penalty,
                              typical_p=self.typical_p,
                              mirostat_mode=self.mirostat_mode,
                              mirostat_tau=self.mirostat_tau,
                              mirostat_eta=self.mirostat_eta,
        )
        print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")

        return self.split_text(response["choices"][0]["text"])
        # return [response["choices"][0]["text"]]

    def chat(self, text, max_tokens=150):
        self.system_prompts = self.system_chat_prompts
        text = text.strip()
        prompt = self.format_prompt(text, history=self.chat_history)
        print(prompt)
        print("Prompt length: " + str(len(prompt)))
        start = time.time()
        response = self.model(prompt=prompt,
                              stop=self.template["end_token"],
                              echo=False,
                              logit_bias=self.biases if self.biases is not None else None,
                              max_tokens=max_tokens,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              min_p=self.min_p,
                              top_k=self.top_k,
                              frequency_penalty=self.frequency_penalty,
                              presence_penalty=self.presence_penalty,
                              repeat_penalty=self.repeat_penalty,
                              typical_p=self.typical_p,
                              mirostat_mode=self.mirostat_mode,
                              mirostat_tau=self.mirostat_tau,
                              mirostat_eta=self.mirostat_eta,
        )
        print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")
        
        response = remove_emojis(response["choices"][0]["text"])
        response = response.strip()
        self.chat_history.append(text)
        self.chat_history.append(response)

        # print the chat history
        for i in range(0, len(self.chat_history), 2):
            print("input: " + self.chat_history[i])
            print("output: " + self.chat_history[i+1])


        print("Chat history length: " + str(len(self.chat_history)))
        # if chat history is more than 15 pairs, we reset to have more variance
        if len(self.chat_history) > 30:
            print("Resetting chat history")
            self.chat_history = self.chat_init.copy()

        return self.split_text(response)
        # return [response]
    
# this class uses the llmapi.py API to communicate with the llama_cpp
class TextGeneratorAPIClient:
    def __init__(self, port=6969, host="127.0.0.1", url=None) -> None:
        if url is not None:
            self.url = url
        else:
            self.url = f"http://{host}:{port}"

    async def rephrase(self, text):
        data = {"text": text}
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url + "/rephrase", data=data) as r:
                print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")
                if r.status == 200:
                    return (await r.json())["sentences"]
                else:
                    return None
        
    async def chat(self, text):
        data = {"text": text}
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url + "/chat", data=data) as r:
                print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")
                if r.status == 200:
                    return (await r.json())["sentences"]
                else:
                    return None
        
class TextGeneratorAPI:
    def __init__(self, config={"port": 6969, "host": "0.0.0.0", "model_path": "E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf", "format_path": "formats/ChatML.json"}) -> None:
        self.port = config["port"]
        self.host = config["host"]
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.tg = TextGenerator(config=config)
        self.init_api()

    def init_api(self):
        @self.app.route("/helloworld", methods=["GET"])
        def helloworld():
            return jsonify({"message": "Hello World!"})

        @self.app.route("/rephrase", methods=["POST"])
        def rephrase():
            text = request.form.get("text")
            if text is None:
                return jsonify({"error": "No text provided!"})
            sentences = self.tg.rephrase(text)
            return jsonify({"sentences": sentences})

        @self.app.route("/chat", methods=["POST"])
        def chat():
            text = request.form.get("text")
            if text is None:
                return jsonify({"error": "No text provided!"})
            sentences = self.tg.chat(text)
            return jsonify({"sentences": sentences})
        
    def run(self):
        self.app.run(host=self.host, port=self.port, debug=False)