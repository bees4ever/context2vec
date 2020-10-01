#!/usr/bin/env python
"""
Learns context2vec's parametric model
"""
import argparse
import pickle
import time
import sys

import numpy as np
from chainer import cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.serializers as S
import chainer.computational_graph as C
from chainer.optimizer_hooks import GradientClipping

from sentence_reader import SentenceReaderDir
from context2vec.common.context_models import BiLstmContext
from context2vec.common.defs import IN_TO_OUT_UNITS_RATIO, NEGATIVE_SAMPLING_NUM
from context2vec.train.corpus_by_sent_length import read_in_corpus


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', '-i',
                        default=None,
                        help='input corpus directory')
    parser.add_argument('--trimfreq', '-t', default=0, type=int,
                        help='minimum frequency for word in training')
    parser.add_argument('--ns_power', '-p', default=0.75, type=float,
                        help='negative sampling power')
    parser.add_argument('--dropout', '-o', default=0.0, type=float,
                        help='NN dropout')
    parser.add_argument('--wordsfile', '-w',
                        default=None,
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m',
                        default=None,
                        help='model output filename')
    parser.add_argument('--cgfile', '-cg',
                        default=None,
                        help='computational graph output filename (for debug)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=300, type=int,
                        help='number of units (dimensions) of one context word')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--context', '-c', choices=['lstm'],
                        default='lstm',
                        help='context type ("lstm")')
    parser.add_argument('--deep', '-d', choices=['yes', 'no'],
                        default=None,
                        help='use deep NN architecture')
    parser.add_argument('--alpha', '-a', default=0.001, type=float,
                        help='alpha param for Adam, controls the learning rate')
    parser.add_argument('--grad-clip', '-gc', default=None, type=float,
                        help='if specified, clip l2 of the gradient to this value')

    args = parser.parse_args()

    if args.deep == 'yes':
        args.deep = True
    elif args.deep == 'no':
        args.deep = False
    else:
        raise Exception("Invalid deep choice: " + args.deep)

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Context type: {}'.format(args.context))
    print('Deep: {}'.format(args.deep))
    print('Dropout: {}'.format(args.dropout))
    print('Trimfreq: {}'.format(args.trimfreq))
    print('NS Power: {}'.format(args.ns_power))
    print('Alpha: {}'.format(args.alpha))
    print('Grad clip: {}'.format(args.grad_clip))
    print('')

    return args

#TODO: LOWER AS ARG
from context2vec.train.sentence_reader import SentenceReaderDict


import numpy


def cosine(a, b):
    norm1 = numpy.linalg.norm(a)
    norm2 = numpy.linalg.norm(b)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0
    else:
        return a.dot(b) / (norm1 * norm2)


class C2VWV:
    def __init__(self, word2index: dict, vocabs: list, matrix):
        self.matrix = matrix
        self.word2index = word2index
        self.vectors = matrix
        self.vocabs = vocabs

    def __getitem__(self, word):
        return self.matrix[self.vocabs.index(word)]


class Context2Vec:
    def __init__(self):
        self.backend_model = None
        self.target_word_units = None
        self.reader = None


    def __run(self, epoch, optimizer):
        for epoch in range(epoch):
            begin_time = time.time()
            cur_at = begin_time
            word_count = 0
            STATUS_INTERVAL = 1000000
            next_count = STATUS_INTERVAL
            accum_loss = 0.0
            last_accum_loss = 0.0
            last_word_count = 0
            print('epoch: {0}'.format(epoch))

            self.reader.open()
            for sent in self.reader.next_batch():

                self.backend_model.zerograds()
                loss = self.backend_model(sent)
                accum_loss += loss.data
                loss.backward()
                del loss
                optimizer.update()

                word_count += len(sent) * len(sent[0])  # all sents in a batch are the same length
                accum_mean_loss = float(accum_loss) / word_count if accum_loss > 0.0 else 0.0

                if word_count >= next_count:
                    now = time.time()
                    duration = now - cur_at
                    throuput = float((word_count - last_word_count)) / (now - cur_at)
                    cur_mean_loss = (float(accum_loss) - last_accum_loss) / (word_count - last_word_count)
                    print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'.format(
                        word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                    next_count += STATUS_INTERVAL
                    cur_at = now
                    last_accum_loss = float(accum_loss)
                    last_word_count = word_count

            print('accum words per epoch', word_count, 'accum_loss', accum_loss, 'accum_loss/word', accum_mean_loss)

    def train(self, sentences, trimfreq=0, ns_power=0.75, dropout=0.0, cgfile=None, gpu=-1, unit=300, batchsize=100, epoch=10, deep=True, alpha=0.001, grad_clip=None):
        print('GPU: {}'.format(gpu))
        print('# unit: {}'.format(unit))
        print('Minibatch-size: {}'.format(batchsize))
        print('# epoch: {}'.format(epoch))
        print('Deep: {}'.format(deep))
        print('Dropout: {}'.format(dropout))
        print('Trimfreq: {}'.format(trimfreq))
        print('NS Power: {}'.format(ns_power))
        print('Alpha: {}'.format(alpha))
        print('Grad clip: {}'.format(grad_clip))
        print('')

        context_word_units = unit
        lstm_hidden_units = IN_TO_OUT_UNITS_RATIO * unit
        self.target_word_units = IN_TO_OUT_UNITS_RATIO * unit

        if gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(gpu).use()
        xp = cuda.cupy if gpu >= 0 else np

        prepared_corpus = read_in_corpus(sentences)

        self.reader = SentenceReaderDict(prepared_corpus, trimfreq, batchsize)
        print('n_vocab: %d' % (len(self.reader.word2index) - 3))  # excluding the three special tokens
        print('corpus size: %d' % (self.reader.total_words))

        cs = [self.reader.trimmed_word2count[w] for w in range(len(self.reader.trimmed_word2count))]
        loss_func = L.NegativeSampling(self.target_word_units, cs, NEGATIVE_SAMPLING_NUM, ns_power)

        #args = parse_arguments()
        self.backend_model = BiLstmContext(deep, gpu, self.reader.word2index, context_word_units, lstm_hidden_units, self.target_word_units, loss_func, True, dropout)

        optimizer = O.Adam(alpha=alpha)
        optimizer.setup(self.backend_model)

        if grad_clip:
            optimizer.add_hook(GradientClipping(grad_clip))


        self.__run(epoch, optimizer)


    @property
    def wv(self):
        return C2VWV(self.reader.word2index, list(c2v.reader.word2index.keys()), self.backend_model.loss_func.W.data)

    def __get_bundle(self):
        return {
            'matrix': self.backend_model.loss_func.W.data,
            'word_units': self.target_word_units,
            'index2word': self.reader.index2word,
            'word2index': self.reader.word2index,
            'backend_model': self.backend_model,
            'wv': self.wv
        }


    def save(self, path):
        context2vec_bundle = self.__get_bundle()

        with open(path, 'wb') as file:
            pickle.dump(context2vec_bundle, file)

corpus = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'],
               ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human',
                'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little',
                'disguised', 'or', 'a', 'little', 'mistaken', '.'],
               ['i', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how', 'much',
                'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'],
               ['men', 'have', 'had', 'every', 'advantage', 'of', 'us', 'in', 'telling', 'their', 'own', 'story', '.',
                'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'],
               ['i', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly', 'happy', ';',
                'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way', '.'],
               ['there', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the', 'less',
                'they', 'will', 'do', 'for', 'themselves', '.'],
               ['one', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of', 'the',
                'other', '.']]


if __name__ == "__main__":
    c2v = Context2Vec()
    c2v.train(corpus, epoch=1)
    print(c2v.wv['of'])
    # words_file = params['config_path'] + params['words_file']
    #         model_file = params['config_path'] + params['model_file']
    #         unit = int(params['unit'])
    #         deep = (params['deep'] == 'yes')
    #         drop_ratio = float(params['drop_ratio'])

    # self.w, self.word2index, self.index2word, self.model = self.read_model(params)

    a2 = c2v.backend_model.context2vec(['a', 'man', 'and', 'a', 'woman'], 3)
    a1 = c2v.backend_model.context2vec(['a', 'man', 'and', 'a', 'woman'], 0)

    print(cosine(a1, a1), cosine(a1, a2))

    of_context = c2v.backend_model.context2vec(['a', 'man', 'of', 'a', 'woman'], 2)
    of_context_2 = c2v.backend_model.context2vec(['a', 'part', 'of', 'the', 'world'], 2)
    cosine(c2v.wv['of'], of_context)
    cosine(c2v.wv['of'], of_context_2)

