import glob
list = glob.glob('./spring-framework/**/*.java', recursive=True)
with open("java_path.txt", mode="w") as f:
    for path in list:
        f.write(path+"\n")
    