import torch
from sampling import Sampling
from encoder_networks import FNNEncoder, LSTMEncoder
from reach import Reach
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from pattern_tokenizer import tokenize


######################################################
######################################################
############      BASE CLASS     #####################
######################################################
######################################################

class EncoderBase:

    def __init__(self, data_infile, fasttext_model_path, triplet_margin=0.1):

        self.sampling = Sampling(data_infile, fasttext_model_path)

        self.amount_negative_names = 1
        self.triplet_margin = triplet_margin
        self.anchor_margin = 0

        self.loss_weights = {'semantic_similarity': 1,
                             'contextual': 1,
                             'grounding': 1}

        torch.autograd.set_detect_anomaly(True)

    def preprocess(self, name):
        return ' '.join(tokenize(name)).lower()

    def combined_loss(self, losses):

        collected_losses = []
        for k, v in losses.items():
            if k in self.loss_weights:
                loss = self.loss_weights[k] * v
                collected_losses.append(loss)

        combined_loss = sum(collected_losses)

        return combined_loss

    def pretrained_loss(self, online_batch, pretrained_batch):

        # take the dot product of the outputted reference and original embedding
        online = online_batch / online_batch.norm(dim=1).reshape(-1, 1)
        pretrained = pretrained_batch / pretrained_batch.norm(dim=1).reshape(-1, 1)
        dot_products = torch.stack([torch.mm(x.reshape(1, -1), y.reshape(1, -1).t()) for x, y in zip(
            online, pretrained)], dim=0)
        dot_product = torch.mean(dot_products)

        pretrained_loss = 1 - dot_product + self.anchor_margin
        pretrained_loss = F.relu(pretrained_loss)

        return pretrained_loss

    def triplet_loss(self, positive_distance, negative_distance, override_margin=False, new_margin=0):

        if override_margin:
            triplet_margin = new_margin
        else:
            triplet_margin = self.triplet_margin

        triplet_loss = positive_distance - negative_distance + triplet_margin
        triplet_loss = F.relu(triplet_loss)

        return triplet_loss

    def positive_distance(self, anchor_batch, positive_batch):

        # take the dot product of the outputted reference and synonym embedding
        ref = anchor_batch / anchor_batch.norm(dim=1).reshape(-1, 1)
        syn = positive_batch / positive_batch.norm(dim=1).reshape(-1, 1)
        dot_products = torch.stack([torch.mm(x.reshape(1, -1), y.reshape(1, -1).t()) for x, y in zip(ref, syn)], dim=0)
        dot_product = torch.mean(dot_products)

        positive_distance = 1 - dot_product

        return positive_distance

    def negative_distance(self, anchor_batch, negatives_batch):

        amount_negative = self.amount_negative_names

        # take the negative dot product of the outputted reference and negatives embeddings
        reference_batch = anchor_batch.reshape(-1, 1, negatives_batch.shape[-1])
        ref = reference_batch / reference_batch.norm(dim=2).reshape(-1, 1, 1)
        neg = negatives_batch / negatives_batch.norm(dim=2).reshape(-1, amount_negative, 1)
        dot_products = []
        for x, y in zip(ref, neg):
            dot_product = torch.mm(x, y.t())
            # apply accumulation strategy for single instance
            accumulated_dot_product = dot_product.mean()
            dot_products.append(accumulated_dot_product)
        dot_products = torch.stack(dot_products, dim=0)

        # extract single loss value for entire batch
        dot_product = torch.mean(dot_products)

        negative_distance = 1 - dot_product

        return negative_distance


######################################################
######################################################
############      FNN BASE     #######################
######################################################
######################################################


class BaseFNN(EncoderBase):

    def __init__(self, input_size=300, hidden_size=38400, num_layers=1, nonlinear=True,
                 num_epochs=200, batch_size=64, learning_rate=0.001, dropout_rate=0.5, gpu_index=-1, **kwargs):

        super().__init__(**kwargs)

        # assign device to train on
        if gpu_index == -1:
            self.gpu = None
            self.cuda = False
            self.device = torch.device('cpu')
        else:
            self.gpu = 'cuda:{}'.format(gpu_index)
            self.cuda = True
            self.device = torch.device(self.gpu)

        # initialize model
        self.hidden_size = hidden_size
        self.input_size = input_size  # input embeddings
        self.output_size = self.input_size  # target embeddings to be learned
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.nonlinear = nonlinear

        self.architecture = FNNEncoder
        self.model = self.architecture(self.input_size, self.hidden_size, self.output_size, self.num_layers,
                                       self.dropout_rate, nonlinear=self.nonlinear).to(self.device)

        # assign training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # assign optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # loss cache
        self.loss_cache = defaultdict(dict)

        self.seed = 1993

    def connect_to_gpu(self, gpu_index):
        self.device = torch.device('cuda:{}'.format(gpu_index))
        self.cuda = True
        self.reinitialize_model()

    def connect_to_cpu(self):
        self.device = torch.device('cpu')
        self.cuda = False
        self.reinitialize_model()

    def reinitialize_model(self):
        self.model = self.architecture(self.input_size, self.hidden_size, self.output_size, self.num_layers,
                                       self.dropout_rate, nonlinear=self.nonlinear).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def change_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self, infile):
        self.model.load_state_dict(torch.load(infile,  map_location=self.gpu))
        self.model.eval()

    def save_model(self, outfile):
        torch.save(self.model.state_dict(), outfile)

    def extract_online_dan_embeddings(self, prune=False, normalize=True, verbose=False, provided_names=(),
                                       preprocess=False):

        self.model.eval()

        if provided_names:
            input_items = provided_names
            if preprocess:
                input_items = [self.preprocess(name) for name in input_items]
            embeddings = self.sampling.create_reach_object(input_items)
        else:
            embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        if prune:
            names_to_prune = set(self.sampling.exemplar_to_concept.keys()).union(self.sampling.validation_references.keys())
            embeddings.prune(names_to_prune)

        input_vectors = embeddings.norm_vectors if normalize else embeddings.vectors
        input_items = [x for _, x in sorted(embeddings.indices.items())]

        # batch input items to save up on memory...
        all_embeddings = []
        batch_size = 1000
        for i in tqdm(range(0, len(input_items), batch_size), disable=not verbose):
            input_batch = input_vectors[i:i + batch_size]
            input_tensor = torch.FloatTensor(input_batch).to(self.device)
            online_batch = self.model(input_tensor).detach().cpu().numpy()
            all_embeddings.append(online_batch)
        all_embeddings = np.concatenate(all_embeddings)

        online_embeddings = Reach(all_embeddings, input_items)

        return online_embeddings
