import argparse
import os
import torch
from src import data_factory


def main():
    args = parse_args()
    dataset = data_factory(args)
    dataset_size = len(dataset)
