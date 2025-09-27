from llava.train.train import train
import os

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    os.system("/usr/bin/shutdown")