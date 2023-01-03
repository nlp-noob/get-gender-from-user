import json
import torch
import copy

from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# The pretag model is the best noted in the model_note.md file
# The path to the model is usually in the saved_model folder

hl_text = "\033[7m{}\033[0m"
red_text = "\033[0;31m{}\033[0m"
green_text = "\033[0;32m{}\033[0m"

@dataclass
class CustomerArguments:
    model_name: str = field(
        metadata={
                    "help": 
                    "Please choose the model to pretag."
                    "In the saved_model path"
                 }
    )
    data_to_be_pretagged: str = field(
        metadata={
                    "help": 
                    "Path to the data to be pretagged for further training."
                    "This is an empty data in common case."
                 }
    )
    tagged_data_output_path: str = field(
        metadata={
                    "help": 
                    "Path to save the output data."
                 }
    )
    num_labels: int = field(
        metadata={
                    "help": 
                    "The number of the labels in the model."
                    "Generally, the label num is 2 or 3."
                 }
    )
    max_line_num: int = field(
        metadata={
                    "help": 
                    "the max line num of a input"
                 }
    )
    device: str = field(
        metadata={
                    "help": 
                    "Path to the data to be pretagged for further training."
                 }
    )
    save_label_after_pretag: bool = field(
        default=False,
        metadata={
                    "help": 
                    "Decide to save the pretag label after the pretagging process or not."
                    "The path is depend on the path of the origin data."
                 }
    )
    use_exist_label: bool = field(
        default=False,
        metadata={
                    "help": 
                    "Decide to use the saved label or not."
                 }
    )
    save_label_path: str = field(
        default=None,
        metadata={
                    "help": 
                    "The path of the saved label."
                 }
    )
    check_the_tagged_order: bool = field(
        default=False,
        metadata={
                    "help": 
                    "Decide to check the tagged data or not after the pretagging process."
                 }
    )
    check_the_empty_order: bool = field(
        default=False,
        metadata={
                    "help": 
                    "Decide to check the tagged data or not after the pretagging process."
                 }
    )
    save_every_step: int = field(
        default=None,
        metadata={
                    "help": 
                    "save the data every step while the checking process."
                 }
    )


class PreTagger():
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.device = args.device
        self.config = AutoConfig.from_pretrained(
                                                 self.model_name, 
                                                 num_labels=args.num_labels,
                                                 finetuning_task="sst2",
                                                 cache_dir=None,
                                                 )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=self.config,
        ).to(self.device)
        self.origin_data = None
        self.a_data_input = {}
        self.a_data_label = {}
        self.all_labels = {}
        # the order index that have label after pretagging process
        self.order_index_with_label = []
        self.tagged_data = None

    def _init_a_data_input(self, order_index):
        # a_data_input is just a dict for a specific order
        # every key in this dict is indicated to the specific line in the order
        # the specific line is for a list of input.
        # a line-input is gradually increasing windowsize doc(upward), until the index reach the top.
        
        # clear the input
        self.a_data_input = {}
        an_order = self.origin_data[order_index]
        for line_index, a_line in enumerate(an_order["order"]):
            # only use the user line as the input
            if not a_line[0]:
                continue
            if line_index not in list(self.a_data_input.keys()):
                self.a_data_input[line_index] = []
            orders_to_be_appended = []
            for up_index in range(line_index + 1):
                now_index = line_index - up_index
                orders_to_be_appended.append(an_order["order"][now_index])
                orders_to_be_appended.reverse()
                self.a_data_input[line_index].append(copy.deepcopy(orders_to_be_appended))
                orders_to_be_appended.reverse()

    def _init_a_data_label(self):
        # clear the label
        self.a_data_label = {}
        # this data label is correspond to the data input
        for a_key in list(self.a_data_input.keys()):
            order_doc_list = self.a_data_input[a_key]
            if a_key not in self.a_data_label:
                self.a_data_label[a_key] = []
            for a_line_input in order_doc_list:
                # set brginning label to out of range
                self.a_data_label[a_key].append(3)

    def _create_a_text(self, order_line_list):
        text_list = []
        for order_line in order_line_list:
            character_name = "[USER]" if order_line[0] else "[ADVISOR]"
            line_text =  character_name + " " + order_line[1]
            text_list.append(line_text)
        return " ".join(text_list)
        
    def _check_illegal(self):
        # find out the label with error
        for a_key in list(self.all_labels.keys()):
            for line_key in list(self.all_labels[a_key].keys()):
                if int(line_key) > len(self.origin_data[a_key]["order"]):
                    raise ValueError("Something went wrong in the tagging process, please check it.")

    def _sumarize_for_pretag(self):
        # make a summary for the all_label
        # do the category at the same time
        all_label_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        order_indexs_have_label = []
        for an_order_key in list(self.all_labels.keys()):
            an_order_label = self.all_labels[an_order_key]
            for line_index in list(an_order_label.keys()):
                # test for display
                if 0 in an_order_label[line_index] or 1 in an_order_label[line_index]:
                    # have label
                    if an_order_key not in order_indexs_have_label:
                        order_indexs_have_label.append(an_order_key)
                for a_label in an_order_label[line_index]:
                    all_label_counter[a_label] += 1
        self.order_index_with_label = order_indexs_have_label
        print("**"*20)
        print("Here is the sammary of the all labels")
        print(all_label_counter)
        print("0-{female}")
        print("1-{male}")
        print("2-{unk}")
        print("3-{out of max sequence length}")
        print("this number is for all windowsize input.")
        print("It means that there must be some duplicated line.")
        print("**"*20)
        print("There are {}/{} order have label".format(len(order_indexs_have_label), len(self.origin_data)))
        print("**"*20)

    def _get_pretag_label_with_model(self):
        for an_order_index, an_order in enumerate(self.origin_data):
            print("Processing order:\t{}/{}".format(an_order_index, len(self.origin_data)))
            self._init_a_data_input(an_order_index)
            self._init_a_data_label()
            # predict
            for line_index in list(self.a_data_input.keys()):
                for a_doc_line_list_index, a_doc_line_list in enumerate(self.a_data_input[line_index]):
                    a_doc_text = self._create_a_text(a_doc_line_list)
                    tokenized_sentence = self.tokenizer(a_doc_text, add_special_tokens=True, return_tensors="pt").to(self.args.device)
                    # jump the process while the tokenized sentence is beyond the max lenth
                    if len(tokenized_sentence.input_ids[0]) >= 512:
                        print("in the size of doc: {}, jump the process of the {} line of the {} order".format(len(a_doc_line_list), line_index, an_order_index))
                        break
                    with torch.no_grad():
                        logits = self.model(**tokenized_sentence).logits
                        label_to_tag = logits.argmax(-1).item()
                        self.a_data_label[line_index][a_doc_line_list_index] = label_to_tag
            # Here the one order pretagging process is done
            self.all_labels[an_order_index] = copy.deepcopy(self.a_data_label)
        # save the pretag label
        if self.args.save_label_after_pretag:
            save_path_words = self.args.data_to_be_pretagged.split("/")
            save_path_words[-1] = "labels_of_" + save_path_words[-1]
            save_path = "/".join(save_path_words)
            fout = open(save_path, "w")
            json_str = json.dumps(self.all_labels, indent=2)
            fout.write(json_str)
            fout.close()
            print("The label is written to the path {}".format(save_path))

    def _get_pretag_label_with_file(self):
        label_file_path = self.args.save_label_path
        fin = open(label_file_path, "r")
        # convert str keys to int 
        tmp_all_labels = json.load(fin)
        for order_key in list(tmp_all_labels.keys()):
            self.all_labels[int(order_key)] = {}
            for line_key in list(tmp_all_labels[order_key].keys()):
                self.all_labels[int(order_key)][int(line_key)] = tmp_all_labels[order_key][line_key]
        fin.close()

    def _get_order_indexs_to_be_tagged(self, mode):
        # get the order indexs of the data to be tagged
        if mode == "tagged":
            return self.order_index_with_label
        elif mode == "empty":
            return_indexs = []
            for order_index in range(len(self.origin_data)):
                if order_index not in self.order_index_with_label:
                    return_indexs.append(order_index)
            return return_indexs

    def _exist_scope(self, new_scope, all_scope):
        # there is no meaning to appending a bigger scope for the label
        # this function checks if any scope of the all_scope is the sub-scope of the new_scope
        for a_scope in all_scope:
            if new_scope[0] >= a_scope[0] and new_scope[1] <= a_scope[1]:
                return True
        False

    def _display_an_order_in_a_scope(self, an_order, a_scope, tag_type):
        id2label = {0: "female", 1: "male", 2: "unk"}
        for line_index, a_line in enumerate(an_order["order"]):
            if line_index <= a_scope[0] and line_index >= a_scope[1]:
                a_line[1] = green_text.format(a_line[1])
            character_name = hl_text.format("[USER]") if a_line[0] else "[ADVI]"
            text = str(line_index) + "\t" + character_name + a_line[1]
            print(text)
        print("**"*20)
        print(hl_text.format("The tag is {}".format(tag_type)))
        print(id2label)
        print("**")

    def _get_right_input(self, order_max_len):
        return_scope = []
        fix_type_id = None
        type_list = [0, 1, 2]
        while(True):
            print(hl_text.format("--------------------Input Info-------------------"))
            print("Please input the real type and scope(e.g. 1 5 3 )")
            print("In this example, 1--type, 5 3 -- scope:[5, 3]")
            print("Or input the command")
            print("j--jump current scope (because the pretagging is right).")
            print("q--quit the current scope tagging.")
            print(hl_text.format("--------------------Input Info-------------------"))
            user_input = input()
            if user_input == "j":
                return False, False
            elif user_input == "q":
                return return_scope, fix_type_id
            else:
                new_tag_str_list = user_input.split(" ")
                if len(new_tag_str_list) != 3:
                    input(red_text.format("Wrong input!!! press any key to continue."))
                    continue
                elif (not new_tag_str_list[0].isdigit() or
                      not new_tag_str_list[1].isdigit() or
                      not new_tag_str_list[2].isdigit()):
                    input(red_text.format("Wrong input!!! press any key to continue."))
                    continue
                else:
                    fix_type_id = int(new_tag_str_list[0])
                    input_scope = [int(new_tag_str_list[1]), int(new_tag_str_list[2])]
                    if fix_type_id not in type_list:
                        input(red_text.format("Wrong input!!! press any key to continue."))
                        continue
                    elif input_scope[0] < input_scope[1]:
                        input(red_text.format("Wrong input!!! press any key to continue."))
                        continue 
                    elif (
                          input_scope[0] < 0 or 
                          input_scope[0] > order_max_len or
                          input_scope[1] < 0 or 
                          input_scope[1] > order_max_len
                         ):
                        input(red_text.format("Wrong input!!! press any key to continue."))
                        continue 
                    return_scope.append(input_scope)

    def _fix_a_scope(self, an_order, a_scope, tag_type, an_order_index):
        self._display_an_order_in_a_scope(an_order, a_scope, tag_type)
        new_scope, new_type_id = self._get_right_input(len(an_order["order"]))
        if new_scope == False and new_type_id == False:
            # use the pretagging scope
            self.tagged_data[an_order_index]["gender_label_keyword"][a_scope[0]] = tag_type
            self.tagged_data[an_order_index]["gender_label_keyword_begin_index"][a_scope[0]] = a_scope[1]
        elif len(new_scope) == 0:
            # no tag to change
            pass
        else:
            for a_new_scope in new_scope:
                self.tagged_data[an_order_index]["gender_label_keyword"][a_new_scope[0]] = new_type_id
                self.tagged_data[an_order_index]["gender_label_keyword_begin_index"][a_new_scope[0]] = a_new_scope[1]


    def _fix_an_order(self, an_order_index, all_label_scope, tag_type):
        an_order = self.tagged_data[an_order_index]
        # display info
        if len(all_label_scope) > 0:
            for a_scope in all_label_scope:
                self._fix_a_scope(an_order, a_scope, tag_type, an_order_index)
        else:
            self._fix_a_scope(an_order, [-1, -1], tag_type, an_order_index)
        

    def _fix_orders(self, mode):
        order_indexs_for_fixxing = self._get_order_indexs_to_be_tagged(mode)
        for process_num, an_order_index in enumerate(order_indexs_for_fixxing):
            print("**"*20)
            print("You are fixxing the order {}/{}".format(process_num + 1, len(order_indexs_for_fixxing)))
            print("**"*20)
            an_order = self.origin_data[an_order_index]
            an_order_label = self.all_labels[an_order_index]
            # 从pretag中取出来相应的label的逻辑就是
            # minimum doc_size principle
            all_label_scope = []
            tag_type = 2
            for line_index in list(an_order_label.keys()):
                a_line_label = an_order_label[line_index]
                for up_index, a_tag in enumerate(a_line_label):
                    if a_tag == 0 or a_tag == 1:
                        tag_type = a_tag
                        new_scope = [line_index, line_index-up_index]
                        if not self._exist_scope(new_scope, all_label_scope):
                            all_label_scope.append(new_scope)
                        break
            # fix process
            self._fix_an_order(an_order_index, all_label_scope, tag_type)
            # save after step
            if process_num % self.args.save_every_step == 0:
                fout = open(self.args.tagged_data_output_path, "w")
                json_str = json.dumps(self.tagged_data, indent=2)
                fout.write(json_str)
                fout.close()
        fout = open(self.args.tagged_data_output_path, "w")
        json_str = json.dumps(self.tagged_data, indent=2)
        fout.write(fout)
        fout.close()
                
    def pretag(self):
        data_path = self.args.data_to_be_pretagged
        fin = open(data_path, "r")
        self.origin_data = json.load(fin) 
        self.tagged_data = copy.deepcopy(self.origin_data)
        fin.close()
        if not self.args.use_exist_label:
            self._get_pretag_label_with_model()
        else:
            self._get_pretag_label_with_file()
        # Here the pretag process is done
        # A summary is shown to the user here
        # At the same time, the index of the order with label is saved to self.
        self._check_illegal()
        self._sumarize_for_pretag()
        # check and edit the label manually
        if self.args.check_the_tagged_order:
            self._fix_orders(mode="tagged")
        if self.args.check_the_empty_order:
            self._fix_orders(mode="empty")
            
            
def main():
    parser = HfArgumentParser((CustomerArguments))
    customer_args = parser.parse_args_into_dataclasses()[0]
    model_path = customer_args.model_name
    data_path = customer_args.data_to_be_pretagged
    # use pretagger
    pretagger = PreTagger(customer_args)
    pretagger.pretag()


if __name__ == "__main__":
    main()
