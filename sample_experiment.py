from datetime import datetime


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
            if "num_batches_tracked" not in param_tensor:
                """
                M5 model contains BatchNorm layers that have special parameters
                """
                self.model_params[param_tensor] *= self.amplifying_factor
            else:
                self.model_params[param_tensor] = 0
        
        self.amplifying_factor *= 1.2

        return self.num_train_samples, self.model_params


curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

rayleaf.run_experiment(
    dataset = "speech_commands",
    dataset_dir = "datasets/speech_commands/",
    output_dir=f"output/sample_experiment-{curr_time}/",
    model = "m5",
    num_rounds = 20,
    eval_every = 5,
    ServerType=Server,
    client_types=[(FlippingClient, 1), (AmplifyingClient, 1), (Client, 40)],
    clients_per_round = 5,
    client_lr = 0.06,
    batch_size = 64,
    seed = 0,
    use_val_set = False,
    num_epochs = 5,
    gpus_per_client_cluster=0.1,
    num_client_clusters=2,
    save_model=False,
    notes = f"2 malicious clients"
)
