from dataclasses import dataclass, field
from transformers import (
        HfArgumentParser,
)

import random
import json
import readline
import gender_guesser.detector as gender
import copy
import sklearn

# This Script is used to generate tagged_data based on keywords and names 
# the data_structure of the labels is shown bellow:
# {
#   "gender_label_name": [2,2,2],
#   "gender_label_keyword": [2,2,2],
#   "gender_label_keyword_begin_index": [1,35,7],
# }
# sub-label: 2
# the meaning of the sub-label is provide by the dict: id2label and label2id bellow.
# {0: "female", 1: "male", 2: "unk"}

# words that indicate that the line maybe contain the gender info
KEYWORDS_LIST = ["gender", "female", "male", "Gender", "Female", "Male", "girlfriend", 
                 "boyfriend", " bf ", " gf ", " his ", " her ", "His ", "Her ", " ex "]

id2label = {0: "female", 1: "male", 2: "unk"}
label2id = {"female": 0, "male": 1, "unk": 2}

# This flag is for the continuous tagging.
TAGGED_FLAG = "gender_tagged" # 1 or 0

hl_format = "\033[7m{}\033[0m"
warning_reminder_format = "\033[0;31m{}\033[0m"
ac_reminder_format = "\033[0;32m{}\033[0m" # green
check_line_reminder_format = "\033[0;32m{}\033[0m" # green

# gender guesser object
gender_detector = gender.Detector()

# the util class for tagging the empty data
class DataTagger():
    def __init__(self, display_lines, label2id=label2id):
        self.display_lines = display_lines
        self.label2id = label2id
        self.result = None

    def display_doc(self)
        for a_line in self.display_lines:
            print(a_line)

    def tag(self):
        while(True):
            self.display_doc()
            print(ac_reminder_format.format("**"*20))
            print(ac_reminder_format.format("**"*20))
        

@dataclass
class CustomerArguments:
    mode: str = field(
        default=None,
        metadata={
                    "help": 
                    "Specify the mod which you want to deal the data with."
                    "the avalaible mode list is shown bellow."
                    "1. {fix_data_manually}: Fix the label of the input data by using keyboard."
                    "2. {count_data_tags}: Count the tags of the tagged data."
                 },
    )
    input_file_path: str = field(
        default=None,
        metadata={
                    "help": 
                    "The path to the tagged person data json file."
                    "this data will be used to get the name of the line, and by using the ngender module."
                    "the possible gender line is caught"
                 },
    )
    realtime_save_path: str = field(
        default=None,
        metadata={
                    "help": 
                    "The save path that saving to while tagging."
                 },
    )
    save_every_step: str = field(
        default=None,
        metadata={
                    "help":
                    "save every step."
                 },
    )
    fix_while_count: bool = field(
        default=False,
        metadata={
                    "help":
                    "When you choose the data mode of count_data_tags."
                    "You can specify this choice to decide if you want to fix the data after the process of counting the labels."
                 },
    )
    fix_no_label: bool = field(
        default=False,
        metadata={
                    "help":
                    "When the --fix_while_data is set to True,"
                    "This Option decide to fix the no_label_data or not."
                 },
    )
    fix_ambiguous: bool = field(
        default=False,
        metadata={
                    "help":
                    "When the --fix_while_data is set to True,"
                    "This Option decide to fix the ambiguous_data or not."
                 },
    )
    split_a_valid_dataset: bool = field(
        default=False,
        metadata={
                    "help":
                    "Decide to split out a valid dataset or nor."
                    "Note: A valid dataset is important for evaluation, so it should be tagged strictly in key word and name."
                 },
    )
    splitte_data_path: str = field(
        default=False,
        metadata={
                    "help":
                    "the path that save the splitted data."
                 },
    )
    valid_dataset_len: int = field(
        default=None,
        metadata={
                    "help":
                    "If you decide to split a valid dataset, you should specify the length of the valid dataset."
                 },
    )

    def __post_init__(self):
        self.check_mode_list = ["fix_data_manually", "count_data_tags"]
        if self.mode not in self.check_mode_list:
            print(ac_reminder_format.format("**"*20))
            print(warning_reminder_format.format("The available mode:"))
            print(ac_reminder_format.format("--"*20))
            for a_mode in self.check_mode_list:
                print(ac_reminder_format.format(a_mode))
            print(ac_reminder_format.format("**"*20))
            raise ValueError("You should choose the one a mode from the list shown above")


def init_data(data_to_be_inited):
    inited_data = copy.deepcopy(data_to_be_inited)
    for a_data_index, a_data in enumerate(data_to_be_inited):
        len_labels = len(a_data["label"])
        inited_data[a_data_index]["gender_label_name"] = [2 for x in range(len_labels)]
        inited_data[a_data_index]["gender_label_keyword"] = [2 for x in range(len_labels)]
        inited_data[a_data_index]["gender_label_keyword_begin_index"] = [0 for x in range(len_labels)]
    return inited_data


def get_keyword_in_line_index(line):
    len_line = len(line)
    for word in KEYWORDS_LIST:
        len_word = len(word)
        left_index = 0
        right_index = len_word
        while(right_index < len_line):
            word_piece = line[left_index:right_index]
            if word_piece == word:
                return [left_index, right_index]
            left_index += 1
            right_index += 1
    return False
    
def _format_and_hl_a_word(gender_result, display_label, word):
    len_gender_result = len(gender_result)
    len_display_word = len(display_label)
    if len_gender_result > len_display_word:
        bias = len_gender_result - len_display_word
        display_label = gender_result
        for i in range(bias):
            word = word + " "
    else:
        bias = len_display_word - len_gender_result
        display_label = gender_result
        for i in range(bias):
            display_label = display_label + " "
    return hl_format.format(display_label), hl_format.format(word)
        

def _format_a_display_line_name(display_labels, display_words):
    sign_line = "--"*20
    sign_line = check_line_reminder_format.format(sign_line)
    words_line = "\t\t".join([check_line_reminder_format.format(a_word) for a_word in display_words])
    labels_line = "\t\t".join(display_labels)
    return sign_line + "\n" + words_line + "\n" + labels_line + "\n" + sign_line


def _format_a_display_line_keyword(keyword_line_index, text_line):
    sign_line = "--"*20
    sign_line = check_line_reminder_format.format(sign_line)
    keyword = text_line[keyword_line_index[0]:keyword_line_index[1]]
    display_words = text_line.split(keyword)
    for a_word_index, a_word in enumerate(display_words):
        display_words[a_word_index] = check_line_reminder_format.format(a_word)
    hl_keyword = hl_format.format(keyword)
    return sign_line + "\n" + hl_keyword.join(display_words) + "\n" + sign_line
    

def get_name_gender_in_line_index(line, labels):
    words = line.split(" ")
    display_words = copy.deepcopy(words)

    # save the word len
    display_labels = []
    # init display_labels (generate the same lenth empty word as the origin word)
    for word in words:
        display_labels.append("".join([" " for i in range(len(word))]))

    for a_label in labels:
        # format the display line
        for a_label_index in a_label:
            a_name_piece = words[a_label_index]
            gender_result = gender_detector.get_gender(a_name_piece)
            a_display_label, a_display_word = _format_and_hl_a_word(
                                                                    gender_result, 
                                                                    display_labels[a_label_index],
                                                                    display_words[a_label_index],
                                                                    )
            display_labels[a_label_index] = a_display_label
            display_words[a_label_index] = a_display_word
    display_line = _format_a_display_line_name(display_labels, display_words)
            
    return display_line 


def _display_a_check_item(a_data, line_index, display_line):
    orders = a_data["order"]
    up_context = []
    final_doc = []
    for i in range(len(a_data["order"])):
        up_index = line_index - (i + 1)
        down_index = line_index + (i + 1)
        if up_index >= 0:
            character_name = f"{up_index}\t[USER]\t" if orders[up_index][0] else f"{up_index}\t[ADVI]\t"
            line = orders[up_index][1]
            up_context.append(character_name + line)
    up_context.reverse()
    final_doc.extend(up_context)
    final_doc.append(f"the line index is:\t{line_index}")
    final_doc.append(display_line)
    # show
    for line in final_doc:
        print(line)
 

def fix_data_mannually(data_to_be_fix, args):
    data_to_be_fix = init_data(data_to_be_fix)
    # The list of index for label label
    # [order_index, line_index]
    # e.g: name_gender_indexs = [[2, 3], [5, 2], [11, 7]]
    keyword_gender_indexs = []
    keyword_gender_display_lines = []
    name_gender_indexs = []
    name_gender_display_lines = []

    fixxed_data = copy.deepcopy(data_to_be_fix)
    for a_data_index, a_data in enumerate(data_to_be_fix):
        for a_line_index, a_line in enumerate(a_data["order"]):
            text_line = a_line[1]

            # check keyword
            keyword_line_index = get_keyword_in_line_index(text_line)
            if keyword_line_index != False and a_line[0]:
                display_line = _format_a_display_line_keyword(keyword_line_index, text_line)           
                keyword_gender_display_lines.append(display_line) 
                keyword_gender_indexs.append([a_data_index, a_line_index])

            # check name
            if len(a_data["label"][a_line_index]) > 0:
                display_line = get_name_gender_in_line_index(text_line, a_data["label"][a_line_index])
                name_gender_display_lines.append(display_line)
                name_gender_indexs.append([a_data_index, a_line_index])

    # check the len 
    if (
            len(keyword_gender_indexs) == len(keyword_gender_display_lines) and
            len(name_gender_indexs) == len(name_gender_display_lines)
       ):
        print(ac_reminder_format.format("Check for len success"))

    # fix data mannually
    keyword_label_key = "gender_label_keyword"
    name_label_key = "gender_label_name"

    ## fix the keyword data:
    for process_index, data_pos_index in enumerate(keyword_gender_indexs):
        data_index = data_pos_index[0]
        line_index = data_pos_index[1]
        origin_label_id = fixxed_data[data_index][keyword_label_key][line_index]
        origin_label_name = id2label[origin_label_id]
        display_line = keyword_gender_display_lines[process_index]
        while(True):
            _display_a_check_item(fixxed_data[data_index], line_index, display_line)    
            print("**"*20)
            print("processing keyword data {}/{}".format(process_index + 1, len(keyword_gender_indexs)))
            print("**"*20)
            print("the id map is shown bollow:")
            print(id2label)
            print("origin label is:\t{}".format(warning_reminder_format.format(origin_label_name)))
            print("**"*20)
            user_input = input("please input the id of the gender to tag this line.\n")
            right_ids_list = [0,1,2]
            if not user_input.isdigit():
                print(warning_reminder_format.format("Wrong input, only allow the input type of int."))
                input(warning_reminder_format.format("PLEASE ENTER ANY KEY TO CONTINUE"))
                continue
            else:
                input_id = int(user_input)
                if input_id not in right_ids_list:
                    print(warning_reminder_format.format("Wrong input, OUT OF RANGE."))
                    print("right range bellow:")
                    print(right_ids_list)
                    input(warning_reminder_format.format("PLEASE ENTER ANY KEY TO CONTINUE"))
                    continue
                else:
                    fixxed_data[data_index][keyword_label_key][line_index] = input_id
                    # default is the line_index
                    fixxed_data[data_index]["gender_label_keyword_begin_index"][line_index] = line_index
                    break
        # find context label
        while(True):
            user_input = input("please input the index of the begin line.\n")
            if not user_input.isdigit():
                print(warning_reminder_format.format("Wrong input, only allow the input type of int."))
                input(warning_reminder_format.format("PLEASE ENTER ANY KEY TO CONTINUE"))
            else:
                input_line_index = int(user_input)
                if input_line_index <= line_index and input_line_index >= 0:
                    fixxed_data[data_index]["gender_label_keyword_begin_index"][line_index] = input_line_index
                    break

        if process_index % args.save_every_step == 0:
            fout = open(args.realtime_save_path, "w") 
            json_str = json.dumps(fixxed_data, indent=2)
            fout.write(json_str)
            fout.close()
            print(ac_reminder_format.format("Write Success"))

    ## fix the name data
    for process_index, data_pos_index in enumerate(name_gender_indexs):
        data_index = data_pos_index[0]
        line_index = data_pos_index[1]
        origin_label_id = fixxed_data[data_index][name_label_key][line_index]
        origin_label_name = id2label[origin_label_id]
        display_line = name_gender_display_lines[process_index]
        # find tips from keyword label
        keyword_label_ids = fixxed_data[data_index][keyword_label_key]
        if 1 in keyword_label_ids:
            keyword_tips = id2label[1]
        elif 0 in keyword_label_ids:
            keyword_tips = id2label[0]
        else:
            keyword_tips = id2label[2]

        while(True):
            _display_a_check_item(fixxed_data[data_index], line_index, display_line)    
            print("**"*20)
            print("processing name data {}/{}".format(process_index + 1, len(name_gender_indexs)))
            print("**"*20)
            print("the id map is shown bollow:")
            print(id2label)
            print("origin label is:\t{}".format(warning_reminder_format.format(origin_label_name)))
            print("keyword tips is:\t{}".format(warning_reminder_format.format(keyword_tips)))
            print("**"*20)
            user_input = input("please input the id of the gender to tag this line.\n")
            right_ids_list = [0,1,2]
            if not user_input.isdigit():
                print(warning_reminder_format.format("Wrong input, only allow the input type of int."))
                input(warning_reminder_format.format("PLEASE ENTER ANY KEY TO CONTINUE"))
                continue
            else:
                input_id = int(user_input)
                if input_id not in right_ids_list:
                    print(warning_reminder_format.format("Wrong input, OUT OF RANGE."))
                    print("right range bellow:")
                    print(right_ids_list)
                    input(warning_reminder_format.format("PLEASE ENTER ANY KEY TO CONTINUE"))
                    continue
                else:
                    fixxed_data[data_index][name_label_key][line_index] = input_id
                    break
        if process_index % args.save_every_step == 0:
            fout = open(args.realtime_save_path, "w") 
            json_str = json.dumps(fixxed_data, indent=2)
            fout.write(json_str)
            fout.close()
            print(ac_reminder_format.format("Write Success"))

    fout = open(args.realtime_save_path, "w") 
    json_str = json.dumps(fixxed_data, indent=2)
    fout.write(json_str)
    fout.close()
    print(ac_reminder_format.format("Write Success"))


def hl_keywords_in_line(line):
    new_line = line
    for keyword in KEYWORDS_LIST:
        if keyword in new_line:
            splitted_line_words = new_line.split(keyword)
            new_line = hl_format.format(keyword).join(splitted_line_words)
    return new_line


def fix_data_perfectly(a_data):    
    display_lines_list = []
    for a_line_index, a_line in enumerate(a_data["order"]):
        display_line_text = hl_keywords_in_line(a_line[1])
        character_name = "[USER]" if a_line[0] else "[ADVI]"
        display_lines_list.append(str(a_line_index) + "\t" + character_name + "\t" + display_line_text)
        

    # fix process
    import pdb;pdb.set_trace()
    
    

def count_data_tags(args):
    out_put_path = args.realtime_save_path
    fout = open(out_put_path, "r")
    tagged_data =json.load(fout)

    # keyword label statistic(len) sumary
    len_doc_keyword = {}

    # orders category
    only_keyword_orders = []
    only_name_orders = []
    name_and_keyword_orders = []
    no_label_orders = []

    # ambiguous orders having the label
    ambiguous_orders = []
    
    for an_order in tagged_data:
        all_label_in_an_order = []
        have_name_label_flag = False
        have_keyword_label_flag = False
        labels_name_base = an_order["gender_label_name"]
        labels_keyword_base = an_order["gender_label_keyword"] 
        labels_keyword_begin_line = an_order["gender_label_keyword_begin_index"]

        # name base
        for a_label in labels_name_base:
            if a_label != label2id["unk"]:
                have_name_label_flag = True
                # find ambiguous
                if a_label not in all_label_in_an_order:
                    all_label_in_an_order.append(a_label)

        # keyword
        for now_line_index, a_label in enumerate(labels_keyword_base):
            if a_label != label2id["unk"]:
                have_keyword_label_flag = True
                begin_line_index = labels_keyword_begin_line[now_line_index]
                # find ambiguous
                if a_label not in all_label_in_an_order:
                    all_label_in_an_order.append(a_label)
            
                # count doc len
                a_label_doc_len = now_line_index - begin_line_index + 1
                if a_label_doc_len in list(len_doc_keyword.keys()):
                    len_doc_keyword[a_label_doc_len]["count"] += 1
                else:
                    len_doc_keyword[a_label_doc_len] = {"count": 1, "description": f"the len of the document is {a_label_doc_len}"}

        # category(no duplicate data)
        if len(all_label_in_an_order) > 1:
            ambiguous_orders.append(an_order)
            continue
        if have_name_label_flag and have_keyword_label_flag:
            name_and_keyword_orders.append(an_order)
        elif have_name_label_flag and not have_keyword_label_flag:
            only_name_orders.append(an_order)
        elif not have_name_label_flag and have_keyword_label_flag:
            only_keyword_orders.append(an_order)
        elif not have_name_label_flag and not have_keyword_label_flag:
            no_label_orders.append(an_order) 

    # display
    print(ac_reminder_format.format("**")*20)
    print(ac_reminder_format.format("The category is shown bellow:"))
    print(ac_reminder_format.format("There are {}\t ambiguous orders.".format(len(ambiguous_orders))))
    print(ac_reminder_format.format("There are {}\t name_and_keyword orders.".format(len(name_and_keyword_orders))))
    print(ac_reminder_format.format("There are {}\t only_keyword orders.".format(len(only_keyword_orders))))
    print(ac_reminder_format.format("There are {}\t only_name orders.".format(len(only_name_orders))))
    print(ac_reminder_format.format("There are {}\t no_label orders.".format(len(no_label_orders))))
    print(ac_reminder_format.format("**")*20)
    len_keys = list(len_doc_keyword)
    len_keys.sort()
    for a_key in len_keys:
        print(ac_reminder_format.format("--")*20)
        print(ac_reminder_format.format(len_doc_keyword[a_key]["description"]), end="")
        print(":\t\t"+ac_reminder_format.format(len_doc_keyword[a_key]["count"]))
    print(ac_reminder_format.format("**")*20)

    # split data for first training test
    # to choose the model for pre-train
    orders_for_first_test = []
    orders_for_first_test.extend(only_keyword_orders)
    orders_for_first_test.extend(only_name_orders)
    orders_for_first_test.extend(name_and_keyword_orders)
    print(ac_reminder_format.format("The length of the first training test orders is \t {}".format(len(orders_for_first_test))))
    print(ac_reminder_format.format("**")*20)
    valid_dataset_len = 40
    valid_dataset = orders_for_first_test[:valid_dataset_len]
    train_dataset = orders_for_first_test[valid_dataset_len:]
    valid_out_path = "./data/test_train/valid.json"
    train_out_path = "./data/test_train/train.json"
    vf = open(valid_out_path, "w")
    json_str = json.dumps(valid_dataset, indent=2)
    vf.write(json_str)
    vf.close()
    tf = open(train_out_path, "w")
    json_str = json.dumps(train_dataset, indent=2)
    tf.write(json_str)
    tf.close()
    print(ac_reminder_format.format("Write first test training dataset complete."))

    # fix the data
    # order keys list
    # ['pairNO', 'orderNO', 'order', 'label', 'gender_label_name', 'gender_label_keyword', 'gender_label_keyword_begin_index']
    if args.fix_while_count:
    
        if args.fix_no_label:
            # fix the data without any label
            for a_data in no_label_orders:
                fix_data_perfectly(copy.deepcopy(a_data))
                for a_line_index, a_line in enumerate(a_data["order"]):
                    pass


def main():
    parser = HfArgumentParser((CustomerArguments))
    customer_args = parser.parse_args_into_dataclasses()[0]
    if customer_args.mode == "fix_data_manually":
        fper = open(customer_args.input_file_path, "r")
        per_data = json.load(fper)
        fper.close()
        fix_data_mannually(per_data, customer_args)
    elif customer_args.mode == "count_data_tags":
        count_data_tags(customer_args)


if __name__=="__main__":
    main()
