import re
import sys
import json

catch_text_compile = re.compile(r".*(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).(\d+).*")
pair_line = "==================== pair:"
order_line = "-------------------- order:"
new_line_symbol = " "
order_price_line = "\t order_price"
filter_chinese = True
generate_empty_label = True

def catch_chinese_in_a_text(text):
    for s in text.encode('utf-8').decode('utf-8'):
        if u'\u4e00' <= s <= u'\u9fff':
            return True
    return False

def filter_chinese_order(orders):
    filtered_orders = []
    for order in orders:
        order_have_chinese = False
        for line in order["order"]:
            text = line[1]
            if catch_chinese_in_a_text(text):
                order_have_chinese = True
                break
        if order_have_chinese:
            filter_or_not = False
            while(True):
                print(order["order"])
                choose_order = input("Please input y/n to decide whether to filter the order:")
                if choose_order.isalpha():
                    if choose_order == "y":
                        filter_or_not = True
                        break
                    elif choose_order == "n":
                        filter_or_not = False
                        break
                    else:
                        print("wrong input!!")
                else:
                    print("wrong input!!")
            if not filter_or_not:
                filtered_orders.append(order)
        else:
            filtered_orders.append(order)
    print("**"*20)
    print("the origin len of the orders is: {}".format(len(orders)))
    print("the len of the filtered orders is: {}".format(len(filtered_orders)))

    return filtered_orders


def format_fin(lines):
    new_lines = [] 
    # record the last line
    last_line = ""
    for line in lines:
        if line == "\n" or line == " \n":
            continue
        if catch_text_compile.match(line):
            new_lines.append(line)
        elif(pair_line not in line and
             order_line not in line and
             order_price_line not in line and
             catch_text_compile.match(last_line)):
            sentence = re.findall("\t.*\t(.*)\t:(.*)", last_line)
            if len(sentence) == 0:
                new_lines[-1] = new_lines[-1][:-1] + line
            else:
                new_lines[-1] = new_lines[-1][:-1] + " " + line
        else:
            new_lines.append(line)
        last_line = line
    return new_lines


def get_labels_in_line(label_list, line_index):
    label_in_line = []
    for label in label_list:
        if label[0]-1 == line_index:
            label_in_line.append(label[1:])
    return label_in_line


def split_alnum_word(word):
    pieces_list = []
    a_piece = ""
    special_piece = ""
    for a_char in word:
        if a_char.isalpha():
            if special_piece:
                pieces_list.append(special_piece)
                special_piece = ""
            a_piece += a_char
        else:
            if a_piece:
                pieces_list.append(a_piece)
                a_piece = ""
            special_piece += a_char
    if a_piece:
        pieces_list.append(a_piece)
    if special_piece:
        pieces_list.append(special_piece)
    return pieces_list


def split_special_word(word):
    # 特殊字符需要进行全部分割
    pieces_list = []
    a_normal_piece = ""
    special_pieces = []
    for a_char in word:
        if a_char.isalnum():
            if special_pieces:
                pieces_list.extend(special_pieces)
                special_pieces = []
            a_normal_piece += a_char
        else:
            if a_normal_piece:
                pieces_list.append(a_normal_piece)
                a_normal_piece = ""
            special_pieces.append(a_char)
    if a_normal_piece:
        pieces_list.append(a_normal_piece)
    if len(special_pieces) > 0:
        pieces_list.extend(special_pieces)
    return pieces_list
    

def format_text(text):
    split_text = text.split()

    word_list = []
    for word in split_text:
        # 去除重复空格
        word = word.strip()
        # 字母数字组合才能作为一个词
        if word.isalnum():
            if word.isalpha():
                word_list.append(word)
            elif word.isdigit():
                word_list.append(word)
            else:
                word_list.extend(split_alnum_word(word))
                
        else:
            splitted_word_list = split_special_word(word)
            for splitted_word in splitted_word_list:
                if splitted_word.isalnum():
                    word_list.extend(split_alnum_word(splitted_word))
                else:
                    word_list.append(splitted_word)
    return " ".join(word_list)


def get_data(fin, label_list, name_list, byte_name_list):
    pair_no = -1
    orders = []
    order_no = 0
    an_order = {"pairNO":0,
                "orderNO":order_no,
                "order":[],
                "label":[] if generate_empty_label else label_list[len(orders)]}
    for line in fin:
    
        if not line.strip():
            continue

        if order_line in line and an_order["order"]:
            orders.append(an_order)
            order_no += 1
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":[] if generate_empty_label else label_list[len(orders)]}

        if pair_line in line:
            pair_no += 1

        if pair_line in line and an_order["order"]:
            orders.append(an_order)
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":[] if generate_empty_label else label_list[len(orders)]}
            order_no = 0

        if catch_text_compile.match(line):
            sentence = re.findall("\t.*\t(.*)\t:(.*)", line)
            name = sentence[0][0]
            if name.encode() in byte_name_list or name in name_list:
                name = False
            else:
                name = True
            text = sentence[0][1]
            text = format_text(text)
            text = text.strip()

            an_order["order"].append([name,text])

        if not(pair_line in line 
               or catch_text_compile.match(line) 
               or order_line in line
               or order_price_line in line):
            text = format_text(line)
            an_order["order"][-1][1] += (new_line_symbol+text)

    orders.append(an_order)

    # 把label转换成每行一个
    for order in orders:
        label_list = order["label"]
        order["label"] = []
        for index in range(len(order["order"])):
            order["label"].append(get_labels_in_line(label_list, index))
            
    return orders


def extract_data(input_raw_data, output_raw_data):
    with open("data/advisor_name_byte.json", "r") as bnf:
        name_list = json.load(bnf)
        byte_name_list = []
        for name in name_list:
            byte_name_list.append(name.encode())
    label_list = []
    with open(input_raw_data, 'r') as fin:
        lines = fin.readlines()
        new_lines = format_fin(lines)
        orders = get_data(new_lines, label_list, name_list, byte_name_list)
        if filter_chinese:
            orders = filter_chinese_order(orders)
    json_str = json.dumps(orders, indent=2)
    with open(output_raw_data, "w") as jf: 
        jf.write(json_str)
        print("Write successed.")

    



def main():

    txt_file = sys.argv[1] if len(sys.argv) > 1 else 'eval_data/order.txt'
    
    with open("../data/advisor_name_byte.json", "r") as bnf:
        name_list = json.load(bnf)
        byte_name_list = []
        for name in name_list:
            byte_name_list.append(name.encode())

    if not generate_empty_label:
        with open("eval_data/label_birth.json", "r") as lf:
            label_list = json.load(lf)
    else:
        label_list = []

    with open(txt_file, 'r') as fin:
        orders = get_data(fin, label_list, name_list, byte_name_list)

    json_str = json.dumps(orders, indent=2)
    with open("eval_data/empty_big.json", "w") as jf: 
        jf.write(json_str)
        print("Write successed.")
    

if __name__=="__main__":
    main()

