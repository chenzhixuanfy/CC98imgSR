#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./CocoData/train2017',
                                     './CocoData/val2017'],
                      test_folders=['./data/BSD100',
                                    './data/Set5',
                                    './data/Set14'],
                      min_size=100,
                      output_folder='./data/')
