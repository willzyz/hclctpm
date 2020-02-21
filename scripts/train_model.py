#!/usr/bin/env python

import argparse
from models.rxgy import train


def main(args):
    print("Start training CL model with args: {}".format(args))
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Michelangelo docker agent')
    parser.add_argument('--train_data_path')
    parser.add_argument('--test_data_path')
    parser.add_argument('--s3_key_for_model')

    args, unknown = parser.parse_known_args()
    main(args)
