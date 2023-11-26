from modules.textgenerator import TextGeneratorAPI

if __name__ == "__main__":
    tgapi = TextGeneratorAPI(model_path="E:/text-generation-webui/models/mistral-7b-openorca.Q5_K_M.gguf")
    tgapi.run()