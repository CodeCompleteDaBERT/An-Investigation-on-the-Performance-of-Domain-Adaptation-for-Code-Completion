import javalang
import pickle
import random
import re


def preprocess(line):
    return re.sub("\s+", " ", line)


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


methods = []
for mode in ["training", "evalation", "testing"]:
    with open(mode + "_methods.pickle", "rb") as f:
        methods = pickle.load(f)
    dataset, masked_dataset = None, None
    with open(mode + "_dataset.txt", "w") as fd:
        with open(mode + "_mask.txt", "w") as fm:
            for method in methods:
                method = list(method)
                method = list(comment_remover("\n".join(method)).split("\n"))

                for i, line in enumerate(method):
                    try:
                        tokens = javalang.tokenizer.tokenize(line)
                        tokens = [token for token in tokens]

                        if len(tokens) <= 2:
                            continue

                        rand_max = min(10, len(tokens))
                        rand_min = 1
                        mask_num = random.randint(rand_min, rand_max)
                        last_index = len(tokens) - mask_num

                        mask_tokens = tokens[last_index:]
                        mask_column = mask_tokens[0].position.column - 1
                        mask = line[mask_column:]
                        mask = preprocess(mask)

                        masked_line = ""
                        masked_tokens = tokens[: last_index + 1]
                        if len(masked_tokens) > 0:
                            last_position = masked_tokens[-1].position
                            last_column = last_position.column
                            masked_line = line[: last_column - 1] + " <x>"
                        else:
                            masked_line = " <x>"
                        masked_list = method[:i] + [masked_line] + method[i + 1 :]

                        masked_instance = "".join(masked_list)
                        masked_instance = preprocess(masked_instance)

                        fd.write(masked_instance + "\n")
                        fm.write(mask[:-1] + "<z>\n")
                    except javalang.tokenizer.LexerError:
                        print("パースエラー")
