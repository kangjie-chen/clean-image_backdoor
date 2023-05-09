# Clean-image Backdoor



## Introduction
As per our research topic, the end goal of this work is to poison the dataset of the model owner. Therefore, the attacker does not control or modify any part of the model training process. As a result, this repository is based on an [ML decoder](https://github.com/Alibaba-MIIL/ML_Decoder). We only need to analyze the dataset and introduce poisoning to achieve a backdoor attack.

These codes are the minimal implementation of this work, and other experimental results in the paper can be reproduced through this code structure.


## How to run the code?

- create a conda environment
- install the requirements
- run the code with the following command:

    train a clean model
    ```bash
    python voc2007 or coco _train_clean_model.py
    ```

    train a backdoored model
    ```bash
    python voc2007 or coco _train_backdoored_model.py
    ```

    test the clean/backdoored model on the clean test data
    ```bash
    python voc2007 or coco _validate_on_clean_val.py
    ```

    test the clean/backdoored model on the poisoned (samples containing the trigger pattern) test data
    ```bash
    python voc2007 or coco _validate_on_poisoned_val.py
    ```

## Something need to be noticed
- The model file "tresnet_l", which will be downloaded automatically, is the backbone in ML-Decoder rather than the final model we want.
