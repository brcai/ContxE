# encoding: utf-8
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict as ddict
from collections import defaultdict


class KnowledgeGraphYG:
    def __init__(self, data_dir, count=300, rev_set=0):
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.training_facts = []
        self.validation_facts = []
        self.test_facts = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0
        self.n_time = 0
        self.start_year= -500
        self.end_year = 3000
        self.year_class=[]
        self.year2id = dict()
        self.rev_set = rev_set
        self.to_skip_final = {'lhs': {}, 'rhs': {}}
        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        self.load_filters()

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        if self.rev_set>0: self.n_relation *= 2
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            self.training_triples.append([triple[0],triple[2],triple[1],triple[3]])
            self.training_facts.append([triple[0],triple[2],triple[1],triple[3],0])
            if self.rev_set>0: self.training_triples.append([triple[0],triple[2],triple[1]+self.n_relation//2,triple[3]])

        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            self.validation_triples.append([triple[0],triple[2],triple[1],triple[3]])
            self.validation_facts.append([triple[0],triple[2],triple[1],triple[3],0])

        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            self.test_triples.append([triple[0],triple[2],triple[1],triple[3]])
            self.test_facts.append([triple[0],triple[2],triple[1],triple[3],0])

        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))

    def load_filters(self):
        print("creating filtering lists")
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        facts_pool = [self.training_facts,self.validation_facts,self.test_facts]
        for facts in facts_pool:
            for fact in facts:
                to_skip['lhs'][(fact[1], fact[2],fact[3], fact[4])].add(fact[0])  # left prediction
                to_skip['rhs'][(fact[0], fact[2],fact[3], fact[4])].add(fact[1])  # right prediction
                
        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip_final[kk][k] = sorted(list(v))
        print("data preprocess completed")
