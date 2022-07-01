import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--repo', help='path to repogitory')
args = parser.parse_args()
path = args.repo
list = glob.glob(path + '/**/*.java', recursive=True)
with open("java_path.txt", mode="w") as f:
    for path in list:
        f.write(path+"\n")