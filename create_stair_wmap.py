import json
import argparse
from tqdm import tqdm

def main(args):
    if args.en:
        with open("annotations/captions_train2014.json", 'r') as f:
            train = json.load(f)

        with open("annotations/captions_val2014.json", 'r') as f:
            val = json.load(f)
    else:
        with open("STAIR-captions/stair_captions_v1.2_train_tokenized.json", 'r') as f:
            train = json.load(f)

        with open("STAIR-captions/stair_captions_v1.2_val_tokenized.json", 'r') as f:
            val = json.load(f)

    tokens = []
    desc = "caption" if args.en else "tokenized_caption"
    for dataset in [train,val]:
        for cmeta in tqdm(dataset["annotations"]):
            tokens.extend(list(map(lambda x: x.replace("\n",""),cmeta[desc].split(' '))))

    tokens_set = set(tokens)
    output = {'<pad>' : 0}
    for i, token in enumerate(tokens_set):
        output[token] = i + 1

    for j, meta in enumerate(['<unk>','<start>','<end>']):
        output[meta] = (i + 1) + j + 1

    print(json.dumps(output, indent=4))
    for k,v in output.items():
        assert "\n" not in k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    parser.add_argument('--en', action='store_true')
    args = parser.parse_args()
    main(args)