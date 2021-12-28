# -*- coding: utf-8 -*-

import os
import numpy
import hnumpy as hnp
from hnumpy.config import CompilationConfig
import logging
import sys
import time
from loguru import logger



def Dissimilarity (user: numpy.ndarray, db: numpy.ndarray,):
    """The function to execute in FHE, on the untrusted server"""
    d = 1 - numpy.dot(user, db)
    print("Encrypted dissimilarity = ", d)
    return d



def user_generates_its_key(context):
    """Done by the user on its private and secure device"""
    logger.info(f"Generating keys")
    keys = context.keygen()

    # Public key: can safely be given to anyone, for FHE computation
    public_keys = keys.public_keys
    
    return keys, public_keys



def compile_function(function, min_weight, max_weight, num_weights):
    """Compile once for all the function"""
    logger.info(f"Compiling the function")
    config = CompilationConfig(parameter_optimizer="handselected")

    fhe_function = hnp.compile_fhe(
        function,
        {
            "user":hnp.encrypted_ndarray(
                bounds=(min_weight, max_weight),
                shape=(num_weights,),),
            "db":hnp.encrypted_ndarray(
                bounds=(min_weight, max_weight),
                shape=(num_weights,),
            ),
        },
        config=config,
    )

    return fhe_function

# set-step is completed
# =========================================================================== #
# =========================[                     ]=========================== #
# =========================================================================== #
# regular cycle

# Encryption
def user_picks_input_and_encrypts(user, db, function, function_string, keys, min_weight, max_weight, num_weights):
    """Done by the user on its private and secure device, with its private keys"""

    # Pick an input
    #weigths = numpy.random.uniform(min_weight, max_weight, (num_weights,)) = user
    logger.info(f"    Picking inputs {user}")

    encrypted_user = keys.encrypt(user)
    encrypted_user = keys.encrypt(db)

    # Also, for comparison, we compute here the expected result
    time_start = time.time()
    #clear_result = function(weigths)
    time_end = time.time()

    logger.info(f"\n    Calling {function_string} in clear")
    logger.info(f"    Result in clear: {clear_result}")
    logger.info(f"    Clear computation was done in {time_end - time_start:.2f} seconds")

    return encrypted_user, encrypted_db #, clear_result


# Evaluation
def running_fhe_computation_on_untrusted_server(
    fhe_function, function_string, public_keys, encrypted_user, encrypted_db
):
    """Done on the untrusted server, but still preserves the user's privacy, thanks
    to the FHE properties. Only public keys are used"""
    logger.info(f"\n    Calling {function_string} in FHE")
    logger.info(f"    Encrypted input shape: {encrypted_weights.shape}")

    time_start = time.time()
    encrypted_result = fhe_function.run(public_keys, encrypted_user, encrypted_db)
    time_end = time.time()

    logger.info(f"    Encrypted result shape after FHE computation: {encrypted_result.shape}")
    logger.info(f"    FHE computation was done in {time_end - time_start:.2f} seconds")

    return encrypted_result

# Decryption
def user_decrypts(keys, encrypted_result):
    """Done by the user on its private and secure device, with its private keys"""
    fhe_result = keys.decrypt(encrypted_result)[0]

    logger.info(f"    Decrypted result as computed through the FHE computation: {fhe_result}")

    return fhe_result



