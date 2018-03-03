import tensorflow as tf
import shutil
import os
import sys
from pickle import Unpickler

from model import WordModel
from utils.cli import export_args

def main(args):
    if os.path.exists(args.data_reader_path):
        print 'Loading data reader...'
        with open(args.data_reader_path, 'rb') as f:
            data_reader = Unpickler(f).load()
            print 'Loaded'

            vocab = data_reader.get_vocab()
    else:
        print "Couldn't load vocab"
        sys.exit()


    print 'Init model...'
    model = WordModel(args, vocab)

    export_dir = os.path.join(args.export_dir, str(args.version))
    print 'Exporting trained model to', export_dir
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    inputs_tensor_info = tf.saved_model.utils.build_tensor_info(model.inputs)
    keep_prob_tensor_info = tf.saved_model.utils.build_tensor_info(model.keep_prob)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(model.gen_seq)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': inputs_tensor_info, 'keep_prob': keep_prob_tensor_info},
            outputs={'outputs': outputs_tensor_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        model.sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'prediction': prediction_signature},
        legacy_init_op=legacy_init_op)

    builder.save()
    print 'Done exporting!'


if __name__ == '__main__':
    main(export_args())
