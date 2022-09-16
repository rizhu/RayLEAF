from datetime import datetime


import numpy as np

from sklearn.utils.extmath import randomized_svd


import rayleaf

from rayleaf.entities import Server, Client
from rayleaf.utils.logging_utils import log


NUM_ROUNDS = 100
NUM_CLIENTS = 400
CLIENTS_PER_ROUND = 20
EVAL_EVERY = 10

CLIENT_LR = 0.05
CLIENT_BATCH_SIZE = 64
NUM_EPOCHS_ON_CLIENTS = 10

GPUS_PER_CLIENT_CLUSTER = 0.75
NUM_CLIENT_CLUSTERS = 4

RANK = 16

curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class CompClient(Client):
    """
    Client that compresses the FC1 layer of the FEMNIST CNN using randomized SVD
    """
    def train(self, server_update):
        #   Update client model to server model parameters
        self.model_params = server_update

        #   Compute gradient updates to server model
        grads = self.train_model(compute_grads=True)

        #   Compress FC1 layer
        res = []
        for layer in grads.tensors:
            if RANK > 0 and layer.shape == (2048, 3136):
                layer = layer.detach().numpy()
                U, S, Vt = randomized_svd(layer, n_components=RANK, n_iter="auto", random_state=None)

                res.append((U, S, Vt))
            else:
                res.append(layer)

        #   Communicate compressed model and number of training samples to server
        return {
            "res": res,
            "n": self.num_train_samples
        }


class CompServer(Server):
    def init(self):
        #   Log model parameter shapes
        log("Model parameters", self.model_params.shapes)


    def server_update(self):
        #   Sends server model to participating clients
        return self.model_params


    def update_model(self, client_updates):
        #   Decompress client updates
        grads_decompressed = []

        for update in client_updates:
            grads_compressed = update["res"]
            decompressed = []
            for layer in grads_compressed:
                if isinstance(layer, tuple):
                    U, S, Vt = layer
                    decompressed.append(np.dot(U * S, Vt))
                else:
                    decompressed.append(layer)
            grads_decompressed.append(rayleaf.TensorArray(decompressed))

        #   Compute weighted average of client updates (FedAvg)
        average_grads = 0
        total = 0
        
        for i, update in enumerate(client_updates):
            average_grads += grads_decompressed[i] * update["n"]
            total += update["n"]

        average_grads /= total

        #   Return updated model
        return self.model_params + average_grads


rayleaf.run_experiment(
    dataset = "femnist",
    dataset_dir = "data/femnist/",
    output_dir= f"output/sample_experiment-{curr_time}/",
    model = "cnn",
    num_rounds = NUM_ROUNDS,
    eval_every = EVAL_EVERY,
    ServerType=CompServer,
    client_types=[(CompClient, NUM_CLIENTS)],
    clients_per_round = CLIENTS_PER_ROUND,
    client_lr = CLIENT_LR,
    batch_size = CLIENT_BATCH_SIZE,
    seed = 0,
    use_val_set = False,
    num_epochs = NUM_EPOCHS_ON_CLIENTS,
    gpus_per_client_cluster = GPUS_PER_CLIENT_CLUSTER,
    num_client_clusters = NUM_CLIENT_CLUSTERS,
    save_model = False,
    notes = f"FedAvg with clients using randomized SVD compression during upload"
)
