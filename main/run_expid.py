import sys

sys.path.append('../')
import os
import logging
from fuxictr import datasets
from datetime import datetime
import time
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
import model_zoo
from model_zoo import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/General_config', help='The config directory.')
    parser.add_argument('--expid', type=str, default='MIRRN', help='The model id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')

    # overwrite param
    parser.add_argument('--dataset', type=str, default='taobao', help='The dataset id to run.')
    parser.add_argument('--embedding_dim', type=str, default='16', help='The embedding size.')
    parser.add_argument('--batch_size', type=str, default='256', help='The batch size.')
    args = vars(parser.parse_args())

    # Load params from config files
    config_dir = args['config']
    experiment_id = args['expid']
    dataset_id = args['dataset']
    params = load_config(config_dir, experiment_id, dataset_id)
    params['gpu'] = args['gpu']
    if args['embedding_dim']:
        params['embedding_dim'] = int(args['embedding_dim'])
    if args['batch_size']:
        params['batch_size'] = int(args['batch_size'])

    # set up logger and random seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Get train and validation data generators
    train_gen, valid_gen = RankDataLoader(feature_map,
                                          stage='train',
                                          train_data=params['train_data'],
                                          valid_data=params['valid_data'],
                                          batch_size=params['batch_size'],
                                          data_format=params["data_format"],
                                          shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])

    logging.info('***** Validation evaluation *****')
    model.evaluate(valid_gen)

    logging.info('***** Test evaluation *****')
    test_gen = RankDataLoader(feature_map,
                              stage='test',
                              test_data=params['test_data'],
                              batch_size=params['batch_size'],
                              data_format=params["data_format"],
                              shuffle=False).make_iterator()
    model.evaluate(test_gen)

