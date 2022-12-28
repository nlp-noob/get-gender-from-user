import json
import copy
import readline
# Some data in the previous version have the duplicated punctuation,
# When this kind of data is sent into the tokenizer, the duplicated piece will not be split into two sub-tokens.
# but during the previous tagging process, the duplicated punctuation piece is seen as a complete token.
# This kind of situation leads to the problem of the label index bias.
# This is a fatal error by using this kind of data during trainning process


# The data path you wanna fix
DATA_TO_BE_FIXXED_PATH = "./data/per_json/train0000_01.json"
hl_format = "\033[7m{}\033[0m"
warning_reminder_format = "\033[0;31m{}\033[0m"
ac_reminder_format = "\033[0;32m{}\033[0m"

# The output path is based on the input path


def _get_display_line(new_words, words, new_label, label):

    for a_new_label, a_label in zip(new_label, label):
        for a_new_label_item, a_label_item in zip(a_new_label, a_label):
            new_words[a_new_label_item] = hl_format.format(new_words[a_new_label_item])
            words[a_label_item] = hl_format.format(words[a_label_item])
    # combine lines
    display_line = " ".join(words) + "\t\t" + "-->" + "\t\t" + " ".join(new_words)
    return display_line


def change_line_and_label(line, label):
    new_words = []
    words = line.split(" ")
    
    bias_accum = 0
    bias_map = []

    have_dup_punc_flag = False
    for word_index, word in enumerate(words):
        if word.isalnum():
            new_words.append(word)
            pass
        else:
            len_punc = len(word)
            if len_punc > 1:
                have_dup_punc_flag = True
                for a_char in word:
                    new_words.append(a_char)
                bias_accum += len_punc - 1
            else:
                new_words.append(word)
        bias_map.append(bias_accum)

    new_label = copy.deepcopy(label)
    for a_label_index, a_label in enumerate(new_label):
        for a_label_item_index, a_label_item in enumerate(a_label):
            new_label[a_label_index][a_label_item_index] += bias_map[a_label_item]

    # auto_check
    auto_check_flag = True
    for a_new_label, a_label in zip(new_label, label):
        for a_new_label_item, a_label_item in zip(a_new_label, a_label):
            if new_words[a_new_label_item] != words[a_label_item]:
                print(warning_reminder_format.format("warning: Auto check False"))
                auto_check_flag = False
                break
    display_line = _get_display_line(
                      copy.deepcopy(new_words), 
                      copy.deepcopy(words), 
                      copy.deepcopy(new_label), 
                      copy.deepcopy(label),
                     )
    if auto_check_flag:
        if have_dup_punc_flag:
            print(ac_reminder_format.format("Auto check complete\t"), end="")
            print(display_line)
    else:
        print(display_line)
        input("Please fix the line:")
    return " ".join(new_words), new_label, have_dup_punc_flag


def fix_data(data_to_be_fixxed):
    fixxed_data = copy.deepcopy(data_to_be_fixxed)
    for data_index, a_data in enumerate(fixxed_data):
        for a_label_index, a_label in enumerate(a_data["label"]):
            text_line = a_data["order"][a_label_index][1]
            fixxed_line, fixxed_label, have_dup_punc_flag = change_line_and_label(text_line, a_label)
            if have_dup_punc_flag:
                fixxed_data[data_index]["order"][a_label_index][1] = fixxed_line
                fixxed_data[data_index]["label"][a_label_index] = fixxed_label

    return fixxed_data


def format_output_path(input_path):
    word_list = input_path.split("/")
    word_list[-1] = "dup_fixxed" + word_list[-1]
    return "/".join(word_list)


def main():
    # Get the data
    fper = open(DATA_TO_BE_FIXXED_PATH, "r")
    per_data = json.load(fper)
    fper.close()
    fixxed_data = fix_data(per_data)
    output_path = format_output_path(DATA_TO_BE_FIXXED_PATH)
    json_str = json.dumps(fixxed_data, indent=2)
    fout = open(output_path, "w")
    fout.write(json_str)
    fout.close()
    print(ac_reminder_format.format("**"*20))
    print(ac_reminder_format.format("**"*20))
    print(ac_reminder_format.format("Write succeed"))
    print(ac_reminder_format.format("**"*20))
    print(ac_reminder_format.format("**"*20))


if __name__=="__main__":
    main()


