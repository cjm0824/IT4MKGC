import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='auto')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/LLaVA/llava/train/checkpoints/llava-v1.5-7b-task-lora")
    parser.add_argument("--model-base", type=str, default="/root/autodl-tmp/llava-v1.5-7b")
    parser.add_argument("--save-model-path", type=str, default="/root/autodl-tmp/LLaVA/llava/train/checkpoints/result_model_checkpoint")

    args = parser.parse_args()

    merge_lora(args)
