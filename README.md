# RayLEAF: A Benchmark for Federated Learning

RayLEAF is a framework for implementing and deploying federated learning experiments. Originally forked from [https://github.com/TalwalkarLab/leaf](LEAF), RayLEAF has been reimplemented from the ground up using PyTorch and Ray to enable high scalability and optimal resource usage and parallelization. To get started, install the package from PyPI
```
pip install rayleaf
```
From there, check out `sample_experiment.py` to see the APIs and workflow. Finally, the source code is open source on GitHub at [https://github.com/rizhu/rayleaf-source](https://github.com/rizhu/rayleaf-source).

## Resources
### LEAF: Benchmark datasets for federated learning
  * **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
  * **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
  We use the LEAF dataset as a benchmarking for our framework, the LEAF dataset contains several supervised ML tasks (including classifcation and predictions) to evaluate the performance of federated learning techniques. 

### Ray
  * **Homepage:** [ray-project](https://github.com/ray-project/ray) is adopted for its universal API for building distributed applications.
We use the  Ray framework for our implementation of distributed framework.

## Datasets

1. FEMNIST

  * **Overview:** Image Dataset
  * **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
  * **Task:** Image Classification

2. Sentiment140

  * **Overview:** Text Dataset of Tweets
  * **Details** 660120 users
  * **Task:** Sentiment Analysis

3. Shakespeare

  * **Overview:** Text Dataset of Shakespeare Dialogues
  * **Details:** 1129 users (reduced to 660 with our choice of sequence length. See [bug](https://github.com/TalwalkarLab/leaf/issues/19).)
  * **Task:** Next-Character Prediction

4. Celeba

  * **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * **Details:** 9343 users (we exclude celebrities with less than 5 images)
  * **Task:** Image Classification (Smiling vs. Not smiling)

5. Synthetic Dataset

  * **Overview:** We propose a process to generate synthetic, challenging federated datasets. The high-level goal is to create devices whose true models are device-dependant. To see a description of the whole generative process, please refer to the paper
  * **Details:** The user can customize the number of devices, the number of classes and the number of dimensions, among others
  * **Task:** Classification

6. Reddit

  * **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
  * **Details:** 1,660,820 users with a total of 56,587,343 comments. 
  * **Task:** Next-word Prediction.

7. Speech Commands

  * **Overview:** We federate the speech data released by [Google](https://www.tensorflow.org/datasets/catalog/speech_commands).
  * **Details:** 2000 users. 
  * **Task:** Audio Classification.

## Notes

- Install the libraries listed in ```requirements.txt```
    - i.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
    - in MacOS check if ```wget``` is installed and working
