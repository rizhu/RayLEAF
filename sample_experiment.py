import rayleaf
from rayleaf.entities import Server, Client


class FlippingClient(Client):
    """
    Malicious client that communicates flipped weights.
    """
    def train(self):
        self.train_model()

        for param_tensor, layer in self.model_params.items():
            self.model_params[param_tensor] *= -1

        return self.num_train_samples, self.model_params


class AmplifyingClient(Client):
    """
    Malicious client that gradually makes its own weights larger.
    """
    def init(self):
        self.amplifying_factor = 1.2

    def train(self):
        self.train_model()

        for param_tensor, layer in self.model_params.items():
            self.model_params[param_tensor] *= self.amplifying_factor
        
        self.amplifying_factor *= 1.2

        return self.num_train_samples, self.model_params


rayleaf.run_experiment(
    dataset = "femnist",
    dataset_dir = "data/femnist/",
    output_dir="output/sample/",
    model = "cnn",
    num_rounds = 20,
    eval_every = 10,
    ServerType=Server,
    client_types=[(FlippingClient, 2), (AmplifyingClient, 2), (Client, -1)],
    clients_per_round = 20,
    client_lr = 0.06,
    batch_size = 64,
    seed = 0,
    use_val_set = False,
    num_epochs = 5,
    gpus_per_client_cluster=0.6,
    num_client_clusters=7,
    save_model=False
)
