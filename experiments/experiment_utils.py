from spex.support_recovery import support_recovery
from spex.utils import mobius_to_fourier, fit_regression, bin_vecs_low_order
import numpy as np
import shapiq
import lime.lime_tabular
import warnings
import pyrootutils
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from itertools import product
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from itertools import product
import logging
import math
import numpy as np
import torch.nn as nn

warnings.filterwarnings("ignore")

SAMPLER_DICT = {
    "spex_hard": "spex",
    "spex_soft": "spex",
    "shapley": "dummy",  # will use sampling from Shap-IQ later on
    "banzhaf": "uniform",
    "lime": "dummy",  # will use sampling from lime later on
    "faith_shapley": "dummy",  # will use sampling from Shap-IQ later on
    "faith_banzhaf": "uniform",
    "shapley_taylor": "dummy",  # will use sampling from Shap-IQ later on,
    "neural_network": "uniform"
}


def spex_hard(signal, b, order=5, **kwargs):
    return support_recovery("hard", signal, b, t=order)


def spex_soft(signal, b, order=5, **kwargs):
    return support_recovery("soft", signal, b, t=order)


def LIME(signal, b, **kwargs):
    training_data = np.zeros((2, signal.n))
    training_data[1, :] = np.ones(signal.n)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data, mode='regression',
                                                            categorical_features=range(signal.n),
                                                            kernel_width=25)  # used in LimeTextExplainer
    lime_values = lime_explainer.explain_instance(np.ones(signal.n), signal.sampling_function,
                                                  num_samples=signal.num_samples(b),
                                                  num_features=signal.n,
                                                  distance_metric='cosine')  # used in LimeTextExplainer
    output = {tuple([0] * signal.n): lime_values.intercept[1]}
    for loc, val in lime_values.local_exp[1]:
        ohe_loc = [0] * signal.n
        ohe_loc[loc] = 1
        output[tuple(ohe_loc)] = val
    return mobius_to_fourier(output)


def shapley(signal, b, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
        index="SV",
        max_order=1
    )
    shapley = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    shapley_dict = {}
    for interaction, ref in shapley.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        shapley_dict[tuple(loc)] = shapley.values[ref]
    return mobius_to_fourier(shapley_dict)


def banzhaf(signal, b, **kwargs):
    return fit_regression('ridge', {'locations': bin_vecs_low_order(signal.n, 1).T}, signal, signal.n, b,
                          fourier_basis=False)[0]


def faith_shapley(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
        index="FSII",
        max_order=order,
    )
    fsii = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    fsii_dict = {}
    for interaction, ref in fsii.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        fsii_dict[tuple(loc)] = fsii.values[ref]
    return mobius_to_fourier(fsii_dict)


def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('ridge', {'locations': bin_vecs_low_order(signal.n, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]


def shapley_taylor(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
        index="STII",
        max_order=order,
    )
    stii = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    stii_dict = {}
    for interaction, ref in stii.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        stii_dict[tuple(loc)] = stii.values[ref]
    return mobius_to_fourier(stii_dict)

def neural_network(signal, b, order=1, num_buckets=8, reg_weight=1e-1, **kwargs):
    coordinates = []
    values = []
    for m in range(len(signal.all_samples)):
        for d in range(len(signal.all_samples[0])):
            for z in range(2 ** b):
                coordinates.append(signal.all_queries[m][d][z])
                values.append(np.real(signal.all_samples[m][d][z]))

    coordinates = np.array(coordinates)
    values = np.array(values)

    class RegressionModel(pl.LightningModule):
        def __init__(self, input_dim, num_buckets, reg_weight=1e-1):
            super().__init__()
            self.num_buckets = num_buckets
            self.reg_weight = reg_weight
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim // 2, input_dim // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim // 4, 1),
            )

        def forward(self, x):
            # if x is not a tensor, convert it to a tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            shifted_x = 2 * x - 1
            return self.model(shifted_x)

        def walsh_hadamard_recursive(self, x):
            """
            Computes the Walsh-Hadamard transform recursively.

            Args:
                x (torch.Tensor): A 1D tensor of size 2^n.

            Returns:
                torch.Tensor: The Walsh-Hadamard transform of x.
            """
            n = x.numel()
            if n == 1:
                return x
            else:
                half_n = n // 2
                x_top = x[:half_n]
                x_bottom = x[half_n:]
                top_part = self.walsh_hadamard_recursive(x_top + x_bottom)
                bottom_part = self.walsh_hadamard_recursive(x_top - x_bottom)
                return torch.cat((top_part, bottom_part))

        def spectral_norm(self):
            binary_vectors = torch.tensor(list(product([0, 1], repeat=self.num_buckets)), dtype=torch.float32, device=self.device)
            binary_matrix = torch.randint(0, 2, (self.num_buckets, self.model[0].in_features), dtype=torch.float32, device=self.device)
            transformed_vectors = torch.matmul(binary_vectors, binary_matrix) % 2
            values = self(transformed_vectors)
            x_tf = self.walsh_hadamard_recursive(values) / 2 ** self.num_buckets
            return x_tf.abs().sum()  # L1 norm of FFT outputs

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.mse_loss(self(x).squeeze(), y)
            if self.num_buckets == 0:
                return loss
            else:
                reg_loss = self.spectral_norm() * self.reg_weight
                return loss + reg_loss
        def validation_step(self, batch, batch_idx):
            val_loss = self.training_step(batch, batch_idx)
            self.log("val_loss", val_loss)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

    # Convert data to tensors
    dataset = torch.utils.data.TensorDataset(torch.tensor(coordinates, dtype=torch.float32),
                                             torch.tensor(values, dtype=torch.float32))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train the model
    model = RegressionModel(input_dim=coordinates.shape[1], num_buckets=num_buckets, reg_weight=reg_weight)

    # Define EarlyStopping callback
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=False)

    # Trainer with EarlyStopping
    trainer = pl.Trainer(max_epochs=500, enable_checkpointing=False, logger=False,
                         enable_model_summary=False, callbacks=[early_stop_callback])

    # Train the model
    trainer.fit(model, dataloader, val_dataloader)

    return model

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)  # Ensures Lightning module is also seeded


class RegressionModel(pl.LightningModule):
    def __init__(self, input_dim, regularization=None, b=8, reg_weight=1e-1):
        super().__init__()
        assert regularization in [None, 'eth', 'conv', 'conv2']
        self.regularization = regularization
        self.b = b
        self.reg_weight = reg_weight
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // 2, input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // 2, 1),
        )
        self.H = None
        self.xor_matrix = None
        self.bin_matrix = None


    def forward(self, x):
        # if x is not a tensor, convert it to a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if self.H is None:
            self.H = self.create_hadamard_matrix(normalize=False)
        shifted_x = 2 * x - 1
        return self.model(shifted_x)

    def create_hadamard_matrix(self, normalize=False):
        '''
        Compute H_n, Hadamard matrix
        '''
        H1 = torch.asarray([[1., 1.], [1., -1.]])
        H = torch.asarray([1.])
        for i in range(self.b):
            H = torch.kron(H, H1)
        if normalize:
            H = (1 / math.sqrt(2 ** self.b)) * H
        return H.to(self.device)

    def create_xor_matrix(self):
        indices = torch.arange(2 ** self.b, dtype=torch.int32)

        # Compute the XOR of every combination using broadcasting
        self.xor_matrix = (indices[:, None] ^ indices[None, :]).to(self.device)

    def create_bin_matrix(self):
        self.bin_matrix = torch.tensor(list(product([0, 1], repeat=self.b)), dtype=torch.float32, device=self.device)

    def walsh_hadamard_recursive(self, x):
        """
        Computes the Walsh-Hadamard transform recursively.

        Args:
            x (torch.Tensor): A 1D tensor of size 2^n.

        Returns:
            torch.Tensor: The Walsh-Hadamard transform of x.
        """
        n = x.numel()
        if n == 1:
            return x
        else:
            half_n = n // 2
            x_top = x[:half_n]
            x_bottom = x[half_n:]
            top_part = self.walsh_hadamard_recursive(x_top + x_bottom)
            bottom_part = self.walsh_hadamard_recursive(x_top - x_bottom)
            return torch.cat((top_part, bottom_part))

    def eth_reg(self):
        if self.bin_matrix is None:
            self.create_bin_matrix()
        random_hash = torch.randint(0, 2, (self.b, self.model[0].in_features), dtype=torch.float32,
                                      device=self.device)
        transformed_vectors = torch.matmul(self.bin_matrix, random_hash) % 2

        values = self(transformed_vectors)
        x_tf = (self.H @ values)  / 2 ** self.b
        return x_tf.abs().sum()  # L1 norm of FFT outputs

    def conv_reg(self):
        if self.bin_matrix is None:
            self.create_bin_matrix()
        random_hash = torch.randint(0, 2, (self.b, self.model[0].in_features), dtype=torch.float32,
                                      device=self.device)
        transformed_vectors = torch.matmul(self.bin_matrix, random_hash) % 2
        values = self(transformed_vectors)

        if self.xor_matrix is None:
            self.create_xor_matrix()

        # Instead, take the convolution of the values
        conv = values.squeeze()[self.xor_matrix] @ values / 2 ** self.b

        x_tf = (self.H @ conv) / 2 ** self.b
        return x_tf.abs().sum()

    def conv_reg2(self):
        if self.bin_matrix is None:
            self.create_bin_matrix()
        random_hash = torch.randint(0, 2, (self.b, self.model[0].in_features), dtype=torch.float32,
                                      device=self.device)
        transformed_vectors = torch.matmul(self.bin_matrix, random_hash) % 2
        conv_values = torch.zeros(2**self.b, dtype=torch.float32, device=self.device)
        y = torch.randint(0, 2, (1000, self.model[0].in_features), dtype=torch.float32,
                          device=self.device)
        y_values = self(y)

        for i in range(2 ** self.b):
            xy = torch.logical_xor(transformed_vectors[i], y).float()
            xy_values = self(xy)

            conv_values[i] = torch.dot(y_values.squeeze(), xy_values.squeeze()) / 1000

        x_tf = (self.H @ conv_values) / 2 ** self.b
        del random_hash, y, xy, conv_values
        torch.cuda.empty_cache()
        return x_tf.abs().sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x).squeeze(), y)
        if not self.regularization:
            return loss
        elif self.regularization == 'eth':
            reg_loss = self.eth_reg() * self.reg_weight
            return loss + reg_loss
        elif self.regularization == 'conv':
            reg_loss = self.conv_reg() * self.reg_weight
            return loss + reg_loss
        else:
            reg_loss = self.conv_reg2() * self.reg_weight
            return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()  # Get the model's predictions

        # Compute R^2
        ss_total = torch.sum((y - torch.mean(y)) ** 2)  # Total sum of squares
        ss_residual = torch.sum((y - y_pred.squeeze()) ** 2)  # Residual sum of squares

        self.log('val_r2', 1-(ss_residual / ss_total))
        return (ss_residual / ss_total)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.05)


def neural_network(signal, b, order = 1, bins=8, reg_weights=[0.01], seed=0):
    coordinates = []
    values = []
    for m in range(len(signal.all_samples)):
        for d in range(len(signal.all_samples[0])):
            for z in range(2 ** b):
                coordinates.append(signal.all_queries[m][d][z])
                values.append(np.real(signal.all_samples[m][d][z]))

    coordinates = np.array(coordinates)
    values = np.array(values)

    seed_everything(seed)
    dataset = torch.utils.data.TensorDataset(torch.tensor(coordinates, dtype=torch.float32),
                                             torch.tensor(values, dtype=torch.float32))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=False)

    val_r2s = []
    trained_models = []
    for reg_weight in reg_weights:
        model = RegressionModel(input_dim=signal.n, regularization="eth",
                                b=bins, reg_weight=reg_weight)

        # Define EarlyStopping callback
        early_stop_callback = EarlyStopping(monitor="val_r2", patience=20, mode="max", verbose=False)

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # Trainer with EarlyStopping
        trainer = pl.Trainer(max_epochs=500,
                             accelerator="auto",
                             devices="auto",
                             enable_checkpointing=False,
                             logger=False,
                             enable_model_summary=True,
                             enable_progress_bar=False,
                             callbacks=[early_stop_callback])

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)
        print(f"Training completed. The trainer ran for {trainer.current_epoch} epochs.")
        model.eval()
        trained_models.append(model)

        val_r2 = 0.0
        with torch.no_grad():
            for val_X, val_y in val_dataloader:
                val_pred = model(val_X).squeeze()
                ss_total = torch.sum((val_y - torch.mean(val_y)) ** 2)  # Total sum of squares
                ss_residual = torch.sum((val_y - val_pred) ** 2)  # Residual sum of squares
                val_r2 += 1 - (ss_residual / ss_total)

        # Average validation loss over the dataset
        val_r2s.append(val_r2)
    print(val_r2s)
    return trained_models[np.argmax(val_r2s)]

def get_ordered_methods(methods, max_order):
    ordered_methods = []
    for method in methods:
        if method in ['shapley', 'banzhaf', 'lime']:
            ordered_methods.append((method, 1))
        elif method in ['faith_banzhaf', 'faith_shapley', 'shapley_taylor']:
            ordered_methods += [(method, order) for order in range(2, max_order + 1)]
        else:
            # spex methods use maximum order 5
            ordered_methods.append((method, 5))
    return ordered_methods


class AlternativeSampler:
    def __init__(self, type, sampling_function, qsft_signal, n):
        self.n = n
        self.num_samples = lambda b: len(qsft_signal.all_samples) * len(qsft_signal.all_samples[0]) * (2 ** b)
        self.sampling_function = sampling_function
        assert type in ["uniform", "dummy"]

        if type == "uniform":
            self.all_queries = []
            self.all_samples = []
            for m in range(len(qsft_signal.all_samples)):
                queries_subsample = []
                samples_subsample = []
                for d in range(len(qsft_signal.all_samples[0])):
                    queries = self.uniform_queries(len(qsft_signal.all_queries[m][d]))
                    queries_subsample.append(queries)
                    samples_subsample.append(sampling_function(queries))
                self.all_queries.append(queries_subsample)
                self.all_samples.append(samples_subsample)

    def uniform_queries(self, num_samples):
        return np.random.choice(2, size=(num_samples, self.n))


def setup_root():
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        cwd=True,
        dotenv=True,
    )
    print(f"Current working directory is set to project root: {root}.")
