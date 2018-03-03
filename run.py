from pickle import Pickler, Unpickler
from os.path import exists

from utils.cli import model_args
from model import WordModel
from utils.data_processing import DataReader

def run(args):
    # TODO: save the results of processing data for faster inference load
    if exists(args.data_reader_path):
        print 'Loading data reader...'
        with open(args.data_reader_path, 'rb') as f:
            data_reader = Unpickler(f).load()
            print 'Loaded'

            vocab = data_reader.get_vocab()
    else:
        print 'Creating data reader...'
        data_reader = DataReader(args.train_dir)

        vocab = data_reader.get_vocab()

        # Save the data reader
        with open(args.data_reader_path, 'wb') as f:
            Pickler(f).dump(data_reader)

    print 'Init model...'
    model = WordModel(args, vocab)

    if args.inference:
        model.generate(primer=args.primer)
    else:
        global_step = 0
        while global_step < args.max_steps:
            inputs, targets = data_reader.get_train_batch(args.batch_size, args.seq_len)
            global_step = model.train_step(inputs, targets)


if __name__ == '__main__':
    args = model_args()

    run(args)