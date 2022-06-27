import lizard
import re

lines = []
NG_WORD = ["/test/"]


def ng_word_detect(path):
    for word in NG_WORD:
        if path in word:
            return False
    return True


with open("java_path.txt") as f:
    lines = f.readlines()
    lines = [l.replace("\n", "") for l in lines]
filtered = filter(ng_word_detect, lines)
method_contexts = []
for path in filtered:
    print(path + "のメソッド抽出を開始")
    methods = lizard.analyze_file(path).__dict__["function_list"]

    with open(path) as f:
        lines = f.readlines()
        for method in methods:
            d = method.__dict__
            line_count = d["end_line"] - d["start_line"]
            name = d["name"]
            # 1. 3行以下のメソッドを削除
            if line_count <= 3:
                continue
            # 2. コピーされたメソッドを削除（CodeSearchNetの機能）
            # 3. 名前にtestが含まれるメソッドを削除
            if "Test" in name:
                continue
            if "test" in name:
                continue
            if "toString" == name:
                continue
            # 行を抽出
            context = lines[d["start_line"] - 1 : d["end_line"]]
            d["context"] = context

            # 行の前処理
            def preprocess(line):
                return re.sub("\s+", " ", line)

            context = list(map(preprocess, context))
            context[0] = re.sub("^\s+", "", context[0])
            # リストに格納
            method_contexts.append(tuple(context))
    print("--------------------------------")

org_count = len(method_contexts)
method_contexts = list(set(method_contexts))
dlt_count = org_count - len(method_contexts)
print(str(dlt_count) + "個のメソッドを削除しました")

import pickle

with open("methods.pickle", mode="wb") as f:
    pickle.dump(method_contexts, f)
