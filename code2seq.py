from argparse import ArgumentParser
import numpy as np
#%tensorflow_version 1.x
import tensorflow as tf
#import geneticalgorithm as ga
from geneticalgorithm import geneticalgorithm as ga
from config import Config
from interactive_predict import InteractivePredictor
from model import Model

#################################################
def evaluate_each_indiv(model,config):
    print("i am in evaluate_ga$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if config.TRAIN_PATH:
        model.train()
    if args.data_path:
        results, precision, recall, f1, rouge = model.evaluate1()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path:
        model.evaluate(release=True)
    
    model.close_session()
    return f1

#################################################
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    #tf.random.set_seed(args.seed)
    ############################################################
    print(args.debug)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)
    print(config.BATCH_SIZE)
    ###GA
    #varbound=np.array([[0,10]]*3)
    #modelga=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)
    #modelga.run()
    #############
    
    model = Model(config)
    print("eheeeeeeeeeeeeeeeeeeeeeeee\n")
    
    aa=evaluate_each_indiv(model,config)
    print("heyyyyyyyyyyyyyyyyy\n")
    print(aa)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()

