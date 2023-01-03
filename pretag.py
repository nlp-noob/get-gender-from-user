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
        
    def _sumarize_for_pretag(self)

    def pretag(self):
        data_path = self.args.data_to_be_pretagged
        fin = open(data_path, "r")
        self.origin_data = json.load(fin) 
        self.tagged_data = copy.deepcopy(self.origin_data)
        fin.close()
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
                        label_to_tag = logits.argmax(-1)
                        self.a_data_label[line_index][a_doc_line_list_index] = label_to_tag
        # Here the pretag process is done
        # A summary is shown to the user here



            
def main():
    parser = HfArgumentParser((CustomerArguments))
    customer_args = parser.parse_args_into_dataclasses()[0]
    model_path = customer_args.model_name
    data_path = customer_args.data_to_be_pretagged
    # use pretagger
    pretagger = PreTagger(customer_args)
    pretagger.pretag()

    # test
    config  = AutoConfig.from_pretrained(
        "./saved_model/pretag_01/",
        num_laebels=3,
        finetuning_task="sst2",
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./saved_model/pretag_01/",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "./saved_model/pretag_01/",
        config=config,
    ).to(customer_args.device)
    text = "jfeijfiwefjewfweiwfieiwijieiewiefioewiofji"
    tokenized_sentence = tokenizer(text, add_special_tokens=True, return_tensors="pt").to(customer_args.device)
    with torch.no_grad():
        logits = model(**tokenized_sentence).logits
        predicted_text_class_id = logits.argmax(-1).item()
    import pdb;pdb.set_trace()

    
    

if __name__ == "__main__":
    main()
