import json
import random
import time
from collections import defaultdict
from copy import deepcopy
from math import ceil
import numpy as np
import torch
from tqdm import tqdm

from synonym_retrieval import SynonymRetrieval
from encoder_base import BaseFNN


######################################################
######################################################
##################      FNN BNE      #################
######################################################
######################################################


class EncoderDAN(BaseFNN):

    def __init__(self,  proto_dropout_rate=0.5, **kwargs):

        super().__init__(**kwargs)

        self.proto_dropout_rate = proto_dropout_rate

    def proto_dropout(self, synset):

        # SYNONYM DROPOUT for conceptual grounding constraint

        assert 0 <= self.proto_dropout_rate < 1

        if not self.proto_dropout_rate:
            return synset

        numsample = ceil(len(synset) * (1 - self.proto_dropout_rate))
        dropout_synset = random.sample(synset, numsample)

        return dropout_synset

    def batch_step(self, positive_samples_batch, negative_name_samples, normalize=True, train=True):

        losses = {}

        # determine which embeddings to sample everything from that's not an anchor name
        train_vectors = self.sampling.train_embeddings.norm_vectors if normalize else self.sampling.train_embeddings.vectors

        if train:
            anchor_embeddings = self.sampling.train_embeddings
        else:
            anchor_embeddings = self.sampling.validation_embeddings
        prototype_embeddings = self.sampling.pretrained_prototype_embeddings

        if train:
            # set model back to train mode if batchnorm is used
            self.model.train()
            # clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

        batch_len = len(positive_samples_batch)

        ################################################
        #####         FIRST COLLECT DATA        ########
        ################################################

        anchor_name_batch = []
        grounding_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            # anchor
            anchor_name_idx = anchor_embeddings.items[anchor]
            if normalize:
                anchor_vector = anchor_embeddings.norm_vectors[anchor_name_idx]
            else:
                anchor_vector = anchor_embeddings.vectors[anchor_name_idx]
            anchor_name_batch.append(anchor_vector)
            # grounding
            concept_idx = prototype_embeddings.items[concept]
            if normalize:
                concept_vector = prototype_embeddings.norm_vectors[concept_idx]
            else:
                concept_vector = prototype_embeddings.vectors[concept_idx]
            grounding_vector = np.average([concept_vector, anchor_vector], axis=0)
            grounding_batch.append(grounding_vector)
        input_anchor_name_batch = torch.FloatTensor(np.array(anchor_name_batch)).to(self.device).reshape(batch_len,
                                                                                                        self.input_size)
        online_anchor_name_batch = self.model(input_anchor_name_batch)

        if train:
            assert self.model.training

        ################################################
        #####          SEMANTIC SIMILARITY      ########
        ################################################

        positive_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            positive_name_idx = self.sampling.train_embeddings.items[positive]
            positive_vector = train_vectors[positive_name_idx]
            positive_name_batch.append(positive_vector)
        input_positive_name_batch = torch.FloatTensor(np.array(positive_name_batch)).to(self.device).reshape(
            batch_len, self.input_size)
        online_positive_name_batch = self.model(input_positive_name_batch)

        positive_name_distance = self.positive_distance(online_anchor_name_batch, online_positive_name_batch)
        losses['positive_name_distance'] = positive_name_distance

        negative_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            negative_name_vectors = []
            for negative in negative_name_samples[anchor]:
                negative_name_idx = self.sampling.train_embeddings.items[negative]
                negative_vector = train_vectors[negative_name_idx]
                negative_name_vectors.append(negative_vector)
            negative_name_batch += negative_name_vectors
        input_negative_name_batch = torch.FloatTensor(np.array(negative_name_batch)).to(self.device).reshape(
            -1, self.input_size)
        online_negative_name_batch = self.model(input_negative_name_batch).reshape(
            batch_len, self.amount_negative_names, self.output_size)

        negative_name_distance = self.negative_distance(online_anchor_name_batch, online_negative_name_batch)
        triplet_name = self.triplet_loss(positive_name_distance, negative_name_distance)
        losses['negative_name_distance'] = negative_name_distance
        losses['semantic_similarity'] = triplet_name

        ######################################################
        #####          CONTEXTUAL MEANINGFULNESS      ########
        ######################################################

        losses['contextual'] = self.pretrained_loss(online_anchor_name_batch, input_anchor_name_batch)

        ######################################################
        #####          CONCEPTUAL GROUNDING           ########
        ######################################################

        grounding_batch = torch.FloatTensor(np.array(grounding_batch)).to(self.device).reshape(-1, self.input_size)
        losses['grounding'] = self.pretrained_loss(online_anchor_name_batch, grounding_batch)

        ################################################
        #####          MULTI-TASK LOSS          ########
        ################################################

        loss = self.combined_loss(losses)
        losses['loss'] = loss

        if train:
            # getting gradients w.r.t. parameters
            loss.backward()
            # updating parameters
            self.optimizer.step()

        losses = {k: v.item() for k, v in losses.items()}

        return losses

    @staticmethod
    def process_losses(losses):

        avg_losses = defaultdict(list)
        for loss_dict in losses:
            for loss_type, loss in loss_dict.items():
                avg_losses[loss_type].append(loss)
        avg_losses = {k: np.mean(v) for k, v in avg_losses.items()}

        return avg_losses

    def train(self, include_validation=True, stopping_criterion=True, random_neg_sampling=False,
              amount_negative_names=1, reinitialize=False, normalize=True, verbose=True, update_iteration=50, outfile=''):

        if reinitialize:
            self.reinitialize_model()
            self.loss_cache = defaultdict(dict)
            self.stopping_criterion_cache = {}

        self.amount_negative_names = amount_negative_names
        assert self.amount_negative_names

        positive_train_samples = self.sampling.positive_sampling(validation=False)
        positive_validation_samples = self.sampling.positive_sampling(validation=True)

        stopping_criterion_cache = {}
        if stopping_criterion:
            self.num_epochs = 1000

        torch.requires_grad = True
        # iterate over epochs
        start = time.time()
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, disable=True):

            # determine epoch ref
            if reinitialize:
                epoch_ref = epoch
            else:
                epoch_ref = max(self.loss_cache) + 1 if self.loss_cache else 1
            print('Started epoch {}'.format(epoch_ref))

            print('Train negative sampling...')
            embeddings = self.extract_online_dan_embeddings(prune=True, normalize=normalize)
            self.sampling.load_online_negative_embeddings(embeddings, prune=True)
            self.sampling.extract_online_prototype_embeddings()

            references = {anchor: concept for (concept, anchor, positive) in positive_train_samples}
            negative_train_samples = self.sampling.negative_name_sampling(references, online=True, validation=False,
                                                                         amount_negative=self.amount_negative_names,
                                                                         verbose=True,
                                                                         random_sampling = random_neg_sampling)

            # iterate over shuffled batches
            print('Training...')
            train_losses = []
            iteration = 0
            random.shuffle(positive_train_samples)
            for i in tqdm(range(0, len(positive_train_samples), self.batch_size), disable=not verbose):
                batch = positive_train_samples[i: i + self.batch_size]
                train_loss = self.batch_step(batch, negative_name_samples=negative_train_samples, normalize=normalize, train=True)
                train_losses.append(train_loss)
                iteration += 1
                if verbose:
                    if iteration % update_iteration == 0:
                        avg_train_losses = self.process_losses(train_losses)
                        print('Iteration: {}. Average training losses: {}'.format(iteration, avg_train_losses))

            # update training loss
            avg_train_losses = self.process_losses(train_losses)
            self.loss_cache[epoch_ref]['train'] = avg_train_losses
            print('Epoch: {}. Average training losses: {}.'.format(epoch_ref,  avg_train_losses))

            # optionally calculate validation loss
            if include_validation:

                print('Validation negative sampling...')
                embeddings = self.extract_online_dan_embeddings(prune=True, normalize=normalize)
                self.sampling.load_online_negative_embeddings(embeddings, prune=True)
                self.sampling.extract_online_prototype_embeddings()

                references = {anchor: concept for (concept, anchor, positive) in positive_validation_samples}
                negative_validation_samples = self.sampling.negative_name_sampling(references, online=True, validation=True,
                                                                             amount_negative=self.amount_negative_names,
                                                                             verbose=True,
                                                                             random_sampling=random_neg_sampling)

                print('Validating...')
                validation_losses = []
                random.shuffle(positive_validation_samples)
                for i in tqdm(range(0, len(positive_validation_samples), self.batch_size), disable=not verbose):
                    batch = positive_validation_samples[i: i + self.batch_size]
                    validation_loss = self.batch_step(batch, negative_name_samples=negative_validation_samples,
                                                              normalize=normalize, train=False)
                    validation_losses.append(validation_loss)

                avg_validation_losses = self.process_losses(validation_losses)
                self.loss_cache[epoch_ref]['validation'] = avg_validation_losses
                print('Epoch: {}. Validation losses: {}.'.format(epoch_ref, avg_validation_losses))

            # optionally calculate stopping criterion
            print('Calculating validation mAP as stopping criterion...')
            if stopping_criterion:
                validation_mAP = self.stopping_criterion()
                stopping_criterion_cache[epoch_ref] = validation_mAP
                stop, best_checkpoint = self.stop_training(epoch_ref)
                if stop:
                    data = {'losses': self.loss_cache,
                            'stopping_criterion': self.stopping_criterion_cache,
                            'best_checkpoint': best_checkpoint}
                    with open('{}.json'.format(outfile), 'w') as f:
                        json.dump(data, f)
                    return

            # save intermediate results
            if outfile:
                data = {'losses': self.loss_cache,
                        'stopping_criterion': stopping_criterion_cache}
                with open('{}.json'.format(outfile), 'w') as f:
                    json.dump(data, f)
                self.save_model('{}_{}.cpt'.format(outfile, epoch_ref))

            print('-------------------------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------------------------')

        print('Finished training!')
        print('Ran {} epochs. Final average training losses: {}.'.format(
            max(self.loss_cache), self.loss_cache[max(self.loss_cache.keys())]
        ))
        end = time.time()
        print('Training time: {} seconds'.format(round(end-start, 2)))

    def stop_training(self, epoch_ref):
        # returns True if stopping criterion has been fulfilled

        lookback_batch = 10
        if epoch_ref <= lookback_batch:
            return False, None

        sorted_values = sorted(self.stopping_criterion_cache.items())
        lookback = sorted_values[epoch_ref-lookback_batch:epoch_ref+1]
        if lookback[-1][1] <= lookback[0][1]:
            stop = True
            best_checkpoint = sorted_values[np.argmax([x for _, x in sorted_values])][0]
        else:
            stop = False
            best_checkpoint = None

        return stop, best_checkpoint

    def stopping_criterion(self):

        validation_ranking = self.synonym_retrieval_test(validation=True)
        instances, rankings = zip(*validation_ranking)
        mAP = SynonymRetrieval().mean_average_precision(rankings)

        return mAP

    def synonym_retrieval_test(self, validation=False, baseline=False, outfile=''):

        data = self.sampling.data

        ontology = data['ontology']
        if validation:
            test = data['validation']
        else:
            test = data['test']
        test_pairs = {reference: concept for (concept, reference, positive) in test}

        ranker = SynonymRetrieval()
        ranker.load_ontology(ontology)

        if baseline:
            train_embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        else:
            train_embeddings = self.extract_online_dan_embeddings(prune=False, normalize=True)

        test_embeddings = deepcopy(train_embeddings)
        ranker.load_train_vectors_object(train_embeddings)
        ranker.load_test_vectors_object(test_embeddings)

        test_ranking = ranker.synonym_retrieval_test(test_pairs, outfile=outfile)

        return test_ranking

    def synonym_retrieval_train(self, baseline=False, outfile=''):

        data = self.sampling.data

        ontology = data['ontology']
        train = data['train']
        train_pairs = {reference: concept for (concept, reference, positive) in train}

        ranker = SynonymRetrieval()
        ranker.load_ontology(ontology)

        if baseline:
            train_embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        else:
            train_embeddings = self.extract_online_dan_embeddings(prune=False, normalize=True)

        ranker.load_train_vectors_object(train_embeddings)

        train_ranking = ranker.synonym_retrieval_train(train_pairs, outfile=outfile)

        return train_ranking
