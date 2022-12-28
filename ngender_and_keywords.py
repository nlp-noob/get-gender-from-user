import json
import readline
import gender_guesser.detector as gender
import copy

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
KEYWORDS_LIST = ["gender", "female", "male", "Gender", "Female", "Male", "girfriend", 
                 "boyfriend", " bf ", " gf ", " his ", " her ", "His ", "Her ",]

# The path to the tagged person data json file.
# this data will be used to get the name of the line, and by using the ngender module,
# the possible gender line is caught
PER_DATA_PATH = "./data/per_json/dup_fixxedtrain0000_01.json"

# After the process step of the save_every_step bellow,
# the tagged data is saved to the path
REALTIME_SAVE_PATH = "./data/per_json/tagged_train0000_01.json"
save_every_step = 20

id2label = {0: "female", 1: "male", 2: "unk"}
label2id = {"female": 0, "male": 1, "unk": 2}

# This flag is for the continuous tagging.
TAGGED_FLAG = "gender_tagged" # 1 or 0

hl_format = "\033[7m{}\033[0m"
warning_reminder_format = "\033[0;31m{}\033[0m"
ac_reminder_format = "\033[0;32m{}\033[0m" # green
check_line_reminder_format = "\033[0;32m{}\033[0m" # green

# for human understanding during the tagging process
SHOW_CONTEXT_SIZE = 10

# gender guesser object
gender_detector = gender.Detector()
gender_detector_type_list = ['unknown', 'female', 'male', 'mostly_male', 'mostly_female', 'andy']


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
    

def fix_data_mannually(data_to_be_fix):
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

        if process_index % save_every_step == 0:
            fout = open(REALTIME_SAVE_PATH, "w") 
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
        if process_index % save_every_step == 0:
            fout = open(REALTIME_SAVE_PATH, "w") 
            json_str = json.dumps(fixxed_data, indent=2)
            fout.write(json_str)
            fout.close()
            print(ac_reminder_format.format("Write Success"))

    fout = open(REALTIME_SAVE_PATH, "w") 
    json_str = json.dumps(fixxed_data, indent=2)
    fout.write(json_str)
    fout.close()
    print(ac_reminder_format.format("Write Success"))


def main():
    fper = open(PER_DATA_PATH, "r")
    per_data = json.load(fper)
    fper.close()
    fix_data_mannually(per_data)
    pass


if __name__=="__main__":
    main()
