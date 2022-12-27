import json
import readline

# words that indicate that the line maybe contain the gender info
KEYWORDS_LIST = ["gender", "female", "male", "Gender", "Female", "Male", "girfriend", 
                 "boyfriend", " bf ", " gf ", " his ", " her ", "His ", "Her ",]
id2label = {0: "female", 1: "male", -1: "unk"}
label2id = {"female": 0, "male": 1, "unk": -1}
check_mode = "direct"
# by bad_cases / by direct
DATA_PATH = "./data/big_json/empty.json"
OUTPUT_PATH = "./data/big_json/empty_fixxed.json"
SELECTED_OUT_PATH = "./data/big_json/selected_data.json"
BADCASE_FILE = "badcases_train/dslim_bert-large-NER_win3_cosine/bad_case_for_step_00002400.txt"
FORBIDDEN_WORDS = ["er", "ke"]
hl_format = "\033[7m{}\033[0m"
reminder_format = "\033[0;31m{}\033[0m"
show_context_size = 5
save_every_step = 200


def search_words(data_to_fix, words):
    if len(words)==1 and len(words[0])<=1:
        return []
    elif len(words)==1 and words[0] in FORBIDDEN_WORDS:
        return []
    index_list = []
    for order_index, order in enumerate(data_to_fix):
        for line_index, line in enumerate(order["order"]):
            if not line[0]:
                continue
            # words全部都在这行才算
            output_index_flag = True
            for a_word in words:
                if not a_word in line[1]:
                    output_index_flag = False
            if output_index_flag:
                index_list.append([order_index,line_index])
    if len(index_list)>10:
        print(words)
        raise ValueError("There too manny results for tagging plz check if the words above is too short") 
    if len(index_list)==0:
        return False 
    else:
        return index_list


def _line_have_keywords(line):
    line_have_keyword = False
    for keyword in KEYWORDS_LIST:
        if keyword in line:
            line_have_keyword = True
            break
    return line_have_keyword


def get_indexs_for_data_check(data_to_fix):
    result_list = []
    for order_index, order in enumerate(data_to_fix):
        for line_index, line in enumerate(order["order"]):
            if not line[0]:
                continue
            if order["label"] != -1:
                result_list.append([order_index, line_index])
            elif(_line_have_keywords(line[1])):
                result_list.append([order_index, line_index])
    return result_list


def display_wrong_info():
    print("=="*20)
    print("wrong input!!! you should input like below:")
    print("1. directly input gender string: {female}/{male}/{unk}")
    print("2. input the id number for the specific label:")
    print("   1 : male")
    print("   0 : female")
    print("   -1: unknow(unk)")
    input("Please Press ENTER To Reinput.")
    print("=="*20)


def display_order_part(an_order, line_index, label, use_context=True):
    if not use_context:
        begin_index = 0
        end_index = len(an_order) - 1
    else:
        if (line_index - show_context_size) < 0:
            begin_index = 0
        else:
            begin_index = (line_index - show_context_size)
        if (line_index + show_context_size) >= len(an_order):
            end_index = len(an_order) - 1
        else:
            end_index = (line_index + show_context_size)
    print("origin sentence is:")
    for index in range(begin_index, end_index+1):
        head_text = "[USER]:\t\t" if an_order[index][0] else "[ADVISOR]:\t"
        if index == line_index:
            print(head_text + hl_format.format(an_order[index][1]))
        else:
            print(head_text + an_order[index][1])

    print("origin label is:")
    print(hl_format.format(label))
    print(hl_format.format(id2label[label]))


def display_tips():
    print("please enter the gender info you get from this sentence.")
    print("1 :{}\t\t0:{}\t\t-1:{}".format(id2label[1], id2label[0], id2label[-1]))
    print("To get more information, you can just enter \"more\"")
    print("or you can quit by entering \"exit\" or \"quit\"")
    print("Please enter your input below:")
    pass
    

def main():
    with open(DATA_PATH, "r") as jf:
        data_to_fix = json.load(jf)
        jf.close()

    if check_mode=="bad_cases":
        # get bad words:
        bad_words_line_indexs = []
        all_words_pattern = "all_words:\t" 
        with open(BADCASE_FILE, "r") as bf:
            bad_data = bf.readlines()
            for order_index, line in enumerate(bad_data):
                if all_words_pattern in line:
                    all_words = line.strip().split("\t")[1:]
                    line_index = search_words(data_to_fix, all_words)
                    if line_index:
                        bad_words_line_indexs.extend(line_index)
                    elif line_index!=False:
                        continue
                    else:
                        print(all_words)
                        raise ValueError("There is not corresponding wrong words above in the data, please check it manually!!!") 
    else:
        bad_words_line_indexs = get_indexs_for_data_check(data_to_fix)

    quit_flag = False
    selected_order_indexs = []
    # show and fix the bad word index
    now_step = 0
    for bad_fix_progress, bad_word_index in enumerate(bad_words_line_indexs):
        now_step += 1
        order_index = bad_word_index[0]
        line_index = bad_word_index[1]
        label = data_to_fix[order_index]["label"]
        label_fix = label
        while(True):
            input_mode = None
            print("=="*30)
            print("processing data \t{}/{}".format(bad_fix_progress+1, len(bad_words_line_indexs)))
            print("=="*30)
            # number_words_in_sentence(data_to_fix[order_index]["order"][line_index][1], label)
            display_order_part(data_to_fix[order_index]["order"], line_index, label, use_context=True)
            display_tips()
            new_label = input() 

            if not new_label.isalpha() and not new_label.isdigit():
                display_wrong_info()
                continue
            elif new_label.isalpha():
                gender_list = list(label2id.keys())
                if new_label in gender_list:
                    # 得到修改过后的labelid
                    label_fix = label2id[new_label]
                    print(reminder_format.format("you have input:{}:{}".format(label_fix, id2label[label_fix])))
                    break
                elif new_label == "quit" or new_label == "exit":
                    quit_flag = True
                    break
                else:
                    display_wrong_info()
                    continue
            elif new_label.isdigit():
                label_id_list = list(id2label.keys())
                input_id_num = int(new_label)
                if input_id_num in label_id_list:
                    label_fix = input_id_num
                    print(reminder_format.format("you have input:\t{} : {}".format(label_fix, id2label[label_fix])))
                    break
                else:
                    display_wrong_info()
                    continue
                    
            if quit_flag:
                break
        if quit_flag:
            break
        else:
            # change done
            if data_to_fix[order_index]["label"] != label_fix and order_index not in selected_order_indexs:
                # save the selected order for augmentetion
                selected_order_indexs.append(order_index)
            data_to_fix[order_index]["label"] = label_fix
        if now_step % save_every_step == 0:
            selected_data_for_aug = []
            for selected_index in selected_order_indexs:
                selected_data_for_aug.append(data_to_fix[selected_index])
            json_str_aug = json.dumps(selected_data_for_aug, indent=2)
            with open(SELECTED_OUT_PATH, "w") as sfout:
                sfout.write(json_str_aug)
                print("File has been saved to the path: {}".format(SELECTED_OUT_PATH))
                sfout.close()
            json_str = json.dumps(data_to_fix, indent=2)
            with open(OUTPUT_PATH, "w") as fout:
                fout.write(json_str)
                print("File has been saved to the path: {}".format(OUTPUT_PATH))
                fout.close()
    import pdb;pdb.set_trace()
    selected_data_for_aug = []
    for selected_index in selected_order_indexs:
        selected_data_for_aug.append(data_to_fix[selected_index])
    json_str_aug = json.dumps(selected_data_for_aug, indent=2)
    with open(SELECTED_OUT_PATH, "w") as sfout:
        sfout.write(json_str_aug)
        print("File has been saved to the path: {}".format(SELECTED_OUT_PATH))
        sfout.close()
    json_str = json.dumps(data_to_fix, indent=2)
    with open(OUTPUT_PATH, "w") as fout:
        fout.write(json_str)
        print("File has been saved to the path: {}".format(OUTPUT_PATH))
        fout.close()
        

if __name__=="__main__":
    main()

