import re
import sys
from nltk.corpus.reader import SyntaxCorpusReader

def tree_to_sentence(s):
    return re.sub(r"\([\S]* |\)", "", s)

def preprocess(line):
    line = tree_to_sentence(line)
    line = line.lower()  # to lower case
    line = re.sub(r"\d+", "", line)  # remove digits
    line = re.sub(r'[^\w\s]', " ", line)  # remove all non-alphanumeric and non-space characters
    line = re.sub(r"\s+", " ", line).strip()  # remove excess white spaces
    return line

def main():

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    print(input_file_name)
    print(output_file_name)

    # Clear file
    open(output_file_name, 'w').close()

    # Write predictions to file
    with open(output_file_name, 'a') as output_file:
        with open(input_file_name) as input_file:
            for line in input_file:
                if line != "\n":
                    output_file.write(preprocess(line) + "\n")


if __name__ == "__main__":
    main()