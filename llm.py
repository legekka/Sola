from modules.textgenerator import TextGenerator
import time

if __name__ == "__main__":
    tg = TextGenerator(model_path="E:/text-generation-webui/models/tinyllama-1.1b-chat-v0.7.Q5_K_M.gguf")
    start = time.time()
    result = tg.rephrase("Map is now complete. Efficiency bonus granted.  Planet one is a medium-gravity terraformable Water world. The trace gas atmosphere is composed mostly of Carbon dioxide with an average surface temperature of 262 Kelvin.  All worthwhile bodies have been mapped.")
    print(result)

    # print time with 2 decimals in seconds
    print(f"Time: {time.time() - start:.2f}s")