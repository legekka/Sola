from modules.textgenerator import TextGeneratorAPI
import json
import argparse

if __name__ == "__main__":
    # create argparser where -c defines the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", default="config.json")
    args = parser.parse_args()

    # load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    tgapi = TextGeneratorAPI(config)
    tgapi.run()