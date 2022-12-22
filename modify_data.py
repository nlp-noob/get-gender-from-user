from utils.order_extract import extract_data
from utils.tag_manually import terminal_tagging

import random
from typing import Dict, List, Optional, Union, Any
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class CustomArguments:
    """
    Arguments for how to modifiy the data.
    """
    mode: str = field(
        metadata={"help": (
                "you should specify the mode you wanna use with this module."
                "choose from below:"
                "{raw_to_json} : extract the input file in a way to the json dataset to the out put file."
                "{tagging_empty} : tag the data by terminal."
                "{split_raw_data} : To split raw data in to the training piece and the valid piece by specifying the path."
                "{split_data_to_htmls} : split the tagged data to htmls files for filter."
                "{tag_with_specific_indexs} : by using indexs in a file to specific location in data. to fix the data."
                )
        },
    )
    input_file: str = field(
        default=None,
        metadata={"help": "Input file to be operated"},
    )
    output_file: str = field(
        default=None,
        metadata={"help": "Output file to be operated"},
    )
    file_to_be_tagged: str = field(
        default=None,
        metadata={"help": "You have to specify the file to be tagged in the mode of tagging."},
    )
    raw_data_path: str = field(
        default=None,
        metadata={"help": "You have to specify the file to be tagged in the mode of tagging."},
    )
    splitted_train_data_path: str = field(
        default=None,
        metadata={"help": "The out put of the splitted train piece of the raw_data."},
    ) 
    splitted_valid_data_path: str = field(
        default=None,
        metadata={"help": "The out put of the splitted valid piece of the raw_data."},
    ) 
    train_valid_ratio: float = field(
        default=None,
        metadata={"help": "specify the ratio between train data and valid data."},
    ) 
    

def split_data_by_ratio(data, ratio):
    train_data = []
    valid_data = []
    data = json.load(data)
    random.shuffle(data)
    valid_max = int(len(data) * 0.1)
    for a_data in data:
        if len(valid_data) < valid_max:
            valid_data.append(a_data)
        else:
            train_data.append(a_data)
    return train_data, valid_data
    

def main():
    parser = HfArgumentParser((CustomArguments))
    custom_args = parser.parse_args_into_dataclasses()[0]
    mode_list = [
                 "raw_to_json", 
                 "tagging_empty", 
                 "split_data_to_htmls", 
                 "tag_with_specific_indexs", 
                 "split_raw_data",
                ]

    if custom_args.mode not in mode_list:
        print(mode_list)
        raise ValueError("Wrong mode choice!!! you have to choose the mode in the list above.")

    if custom_args.mode == "raw_to_json":
        if custom_args.input_file is not None and custom_args.output_file is not None:
            extract_data(custom_args.input_file, custom_args.output_file)
            print("write success")
        else:
            raise ValueError("you haven't specify the input file or out put file yet, plz fix it in data.sh")
    elif custom_args.mode == "tagging_empty":
        if custom_args.file_to_be_tagged is not None:
            terminal_tagging(custom_args.file_to_be_tagged)
            print("write success")
        else:
            raise ValueError("you haven't specify the file to be tagged yet, plz fix it in data.sh")
    elif custom_args.mode == "split_raw_data":
        rf = open(custom_args.raw_data_path, "r")
        split_ratio = custom_args.train_valid_ratio
        train_data, valid_data  = split_data_by_ratio(rf, split_ratio)
        rf.close()
        train_data_json_str = json.dumps(train_data, indent=2)
        valid_data_json_str = json.dumps(valid_data, indent=2)
    
        tf = open(custom_args.splitted_train_data_path, "w")
        tf.write(train_data_json_str)
        tf.close()
        vf = open(custom_args.splitted_valid_data_path, "w")
        vf.write(valid_data_json_str)
        vf.close()


if __name__=="__main__":
    main()

