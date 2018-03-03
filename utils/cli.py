import os
import argparse

from misc import date_str, get_dir

def model_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--train_dir',
                        help='Directory of train data',
                        default='./data/poetryDB/txt/')
    # parser.add_argument('--test_dir',
    #                     help='Directory of test data',
    #                     default='./data/bitmoji/test')
    parser.add_argument('--save_dir',
                        help='Directory to save logs and model checkpoints',
                        default=os.path.join('.', 'save', date_str()))
    parser.add_argument('--load_path',
                        help='Path of the model checkpoint to load')
    parser.add_argument('--data_reader_path',
                        help='Path to save/load the DataReader object',
                        default=os.path.join('.', 'save', 'reader.pkl'))

    # Model Architecture
    parser.add_argument('--cell_size',
                        help='Minibatch size',
                        default=256,
                        type=int)
    parser.add_argument('--num_layers',
                        help='Minibatch size',
                        default=3,
                        type=int)

    # Hyperparams
    parser.add_argument('--batch_size',
                        help='Minibatch size',
                        default=128,
                        type=int)
    parser.add_argument('--seq_len',
                        help='Sequence length (the number of tokens in each element of the batch)',
                        default=20,
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--lr_decay_steps',
                        help='The number of steps over which to decay by a multiple of lr_decay_rate',
                        default=200,
                        type=int)
    parser.add_argument('--lr_decay_rate',
                        help='The multiple by which to decay the learning rate every lr_decay_steps steps',
                        default=0.9,
                        type=float)
    parser.add_argument('--keep_prob',
                        help='The keep probability for dropout (always 1 for testing)',
                        default=0.5,
                        type=float)

    # Training
    parser.add_argument('--max_steps',
                        help='Max number of steps to train',
                        default=30000,
                        type=int)
    parser.add_argument('--summary_freq',
                        help='Frequency (in steps) with which to write tensorboard summaries',
                        default=100,
                        type=int)
    parser.add_argument('--model_save_freq',
                        help='Frequency (in steps) with which to save the model',
                        default=1000,
                        type=int)
    parser.add_argument('--inference_freq',
                        help='Frequency (in steps) with which to perform inference',
                        default=100,
                        type=int)

    # Inference
    parser.add_argument('--inference',
                        help="Use the model to generate new text.",
                        action='store_true')
    parser.add_argument('--argmax',
                        help="Use argmax to choose the next word, rather than sampling.",
                        action='store_true')
    parser.add_argument('--max_gen_len',
                        help="The maximum number of words to generate.",
                        default=20,
                        type=int)
    parser.add_argument('--primer',
                        help="The priming text to use for inference. Random if not supplied",
                        default=None)


    # System
    parser.add_argument('--gpu',
                        help='Comma separated list of GPU(s) to use.')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    get_dir(args.save_dir)

    return args


def export_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--export_dir',
                        help='Directory to save the data',
                        default='save/serving/')
    parser.add_argument('--load_path',
                        help='Path of the model checkpoint to load',
                        default='save/hal-3layer/model-9001')
    parser.add_argument('--version',
                        help='Version of the model to save',
                        default=0,
                        type=int)
    parser.add_argument('--data_reader_path',
                        help='Path to save/load the DataReader object',
                        default=os.path.join('.', 'save', 'reader.pkl'))

    # Model Architecture
    parser.add_argument('--cell_size',
                        help='Minibatch size',
                        default=256,
                        type=int)
    parser.add_argument('--num_layers',
                        help='Minibatch size',
                        default=3,
                        type=int)

    # Hyperparams
    parser.add_argument('--batch_size',
                        help='Minibatch size',
                        default=128,
                        type=int)
    parser.add_argument('--seq_len',
                        help='Sequence length (the number of tokens in each element of the batch)',
                        default=20,
                        type=int)
    parser.add_argument('--keep_prob',
                        help='The keep probability for dropout (always 1 for testing)',
                        default=1,
                        type=float)
    parser.add_argument('--lr',
                        help='Learning rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--lr_decay_steps',
                        help='The number of steps over which to decay by a multiple of lr_decay_rate',
                        default=200,
                        type=int)
    parser.add_argument('--lr_decay_rate',
                        help='The multiple by which to decay the learning rate every lr_decay_steps steps',
                        default=0.9,
                        type=float)

    # Inference
    parser.add_argument('--argmax',
                        help="Use argmax to choose the next word, rather than sampling.",
                        action='store_true')
    parser.add_argument('--max_gen_len',
                        help="The maximum number of words to generate.",
                        default=20,
                        type=int)

    args = parser.parse_args()

    return args