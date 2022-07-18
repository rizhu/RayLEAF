# RayLEAF: Federated learning with Privacy

RayLeaf is a framework that supports federated learning with differential privacy. 
* [*Federated learning*](https://en.wikipedia.org/wiki/Federated_learning) refers to machine learning (ML) techniques that train models or algorithms across multiple devices/nodes with local data samples, without communicating them. This would allow us to train the learning model or algorithms moreefficiently, by leveraging the computational resources from multiple devices/nodes.
* [*Differential privacy*](https://en.wikipedia.org/wiki/Differential_privacy) protects the local data by constraining on the algorithms access and disclosure of private information that can be used to identify individuals. This will protect the individual devices/nodes from being identified in the training process.

The motivating question that we ask is: Can we apply differential privacy to local data samples, while still benefit from the federated learning setting? We want to protect individual's privacy, but gain efficiency from the the distributed setting.


## Resources
### LEAF: Benchmark datasets for federated learning
  * **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
  * **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
  We use the LEAF dataset as a benchmarking for our framework, the LEAF dataset contains several supervised ML tasks (including classifcation and predictions) to evaluate the performance of federated learning techniques. 

### Ray
  * **Homepage:** [ray-project](https://github.com/ray-project/ray) is adopted for its universal API for building distributed applications.
We use the  Ray framework for our implementation of distributed framework. For each local node, our algorithm passes the gradient to the serve for model training.

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

## Notes

- Install the libraries listed in ```requirements.txt```
    - i.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
    - in MacOS check if ```wget``` is installed and working
- ```models``` directory contains instructions on running baseline reference implementations

## Experiments
- Malicious client that communicates flipped weights.
- Malicious client that gradually makes its own weights larger.
