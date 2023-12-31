import llama_cpp 
import random
import time

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import aiohttp

import re

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
    def __init__(self, model_path="E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf"):
        self.model = llama_cpp.Llama(model_path=model_path,
                        n_ctx=8192,
                        n_batch=512,
                        n_threads=1,
                        n_gpu_layers=128,
                        #verbose=False
                        )
        with open("formats/ChatML.txt", "r") as f:
            self.template = f.read()
        with open("prompts/system_rephrase.txt", "r") as f:
            self.system_rephrase_prompts = f.read().splitlines()
        with open("prompts/system_chat.txt", "r") as f:
            self.system_chat_prompts = f.read().splitlines()

        self.biases = {
            "315": -10,   # " I"
            "630": -0.5,  # " she"
            "478": -0.5,  # " we"
            "816": -0.5,  # " We"
            "985": -0.5,  # " She"
            "3489": -0.5, # " Our"
        }
        self.chat_init = ["Initializing ship systems, preparing to launch.", "Sola is getting all the spaceship systems ready, and preparing for a big, exciting launch!"]
        self.chat_history = self.chat_init.copy() 

    def split_text(self, text):
        # splits text into multiple sentences based on punctuation ".", "!", "?", ";", ":", and newline "\n"
        # it's important to check the next character after the punctuation, because we don't want to split on for example numbers

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
                        if len(sentences[i].split(" ")) < 5:
                            sentences[i+1] = sentences[i] + sentences[i+1]
                        else:
                            sections.append(sentences[i])
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
        prompt = self.template.format(system=random.choice(self.system_rephrase_prompts), prompt=text)
        start = time.time()
        response = self.model(prompt=prompt,
                              stop="<|im_end|>",
                              echo=False,
                              logit_bias=self.biases,
                              max_tokens=max_tokens,
                              temperature=0.7,
                              top_p=0.9,
                              min_p=0,
                              top_k=20,
                              frequency_penalty=0,
                              presence_penalty=0,
                              repeat_penalty=1.15,
                              typical_p=1,
        )
        print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")

        return self.split_text(response["choices"][0]["text"])
        # return [response["choices"][0]["text"]]

    def chat(self, text, max_tokens=150):
        prompt = self.template.format(system=random.choice(self.system_chat_prompts), prompt=text)
        # we want to split the prompt at the first <|im_end|> occurence, but only the first one
        system_prompt = prompt.split("<|im_end|>\n", 1)[0] + "<|im_end|>\n"
        user_prompt = prompt.split("<|im_end|>\n", 1)[1]
        prompt = system_prompt
        if len(self.chat_history) > 0:
            # loop through the chat history by pairs
            for i in range(0, len(self.chat_history), 2):
                prompt += "<|im_start|>user\n" + self.chat_history[i] + "<|im_end|>\n"
                prompt += "<|im_start|>assistant\n" + self.chat_history[i+1] + "<|im_end|>\n"
        
        prompt += user_prompt
        print(prompt)
        print("Prompt length: " + str(len(prompt)))
        start = time.time()
        response = self.model(prompt=prompt,
                              stop="<|im_end|>",
                              echo=False,
                              logit_bias=self.biases,
                              max_tokens=max_tokens,
                              temperature=0.7,
                              top_p=0.9,
                              min_p=0,
                              top_k=20,
                              frequency_penalty=0,
                              presence_penalty=0,
                              repeat_penalty=1.15,
                              typical_p=1,
        )
        print("LlamaCpp API time: " + str(round(time.time() - start, 2)) + "s")
        
        response = remove_emojis(response["choices"][0]["text"])
        self.chat_history.append(text)
        self.chat_history.append(response)

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
    def __init__(self, port=6969, host="0.0.0.0", model_path = "E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf") -> None:
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.tg = TextGenerator(model_path=model_path)
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