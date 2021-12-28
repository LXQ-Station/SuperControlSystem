# -*- coding: utf-8 -*-

import os
import numpy
import hnumpy as hnp
from hnumpy.config import CompilationConfig
import logging
import sys
import time
from loguru import logger
from util import *
import pickle

# Settings
min_weight = -1
max_weight = 1
num_weights = 128
function = Dissimilarity
function_string = "Dissimilarity"

if __name__ == "__main__":

    fhe_function = compile_function(function, min_weight, max_weight, num_weights)
    context = fhe_function.create_context()

    keys, public_keys = user_generates_its_key(context)

    for i in range(2):

        X = numpy.load("./featureG.npy")
        X = numpy.squeeze(X)
        Y = numpy.load("./featureL4.npy")
        Y = numpy.squeeze(Y)
        print(X)
        logger.info(f"\nRunning {i}-th test")
        result = fhe_function.encrypt_and_run(keys, X, Y)
        print("LOGIN? => ",result)
   
        '''
        encrypted_weights = user_picks_input_and_encrypts(X, Y
            ,function, function_string, keys, min_weight, max_weight, num_weights
        )

    # 3 - This is the FHE execution, done on the untrusted server
        encrypted_result = running_fhe_computation_on_untrusted_server(
            fhe_function, function_string, public_keys, encrypted_weights
        )
        '''

        '''
    # 5 - Finally, for the check and demo, comparing the results. Remark that
    # in a real product, once it is known that FHE results are precise
    # enough
        diff = numpy.abs(fhe_result - clear_result)
        ratio = diff / numpy.max(clear_result)

        logger.info(
            f"\n    Difference between computation in clear and in FHE (expected to be as small as possible): {diff}"
        )
        logger.info(f"    Ratio of difference (expected to be as small as possible): {100 * ratio:.2f} %")
        '''

