# As usual, a bit of setup
from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url
import nltk

# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(pca_features=True)
# med_data = load_coco_data(max_train=50000)
med_data = data
big_lstm_model = None

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

################################################################################

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))

################################################################################

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ') 
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ') 
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

def evaluate_model(model):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.
    """
    BLEUscores = {}
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(med_data, split=split, batch_size=1000)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))

################################################################################

def main():
    args = sys.argv[1:]

    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(231)
    
    big_data = load_coco_data(max_train=8000)
    #big_data = data

    if (len(args) > 0 and args[0] == '--read'):
        with open('old_model.pkl', 'rb') as handle:
            big_lstm_model = pickle.load(handle)
    else:
        big_lstm_model = CaptioningRNN(
            cell_type='lstm',
            word_to_idx=data['word_to_idx'],
            input_dim=data['train_features'].shape[1],
            hidden_dim=1024,
            wordvec_dim=512,
            dtype=np.float32,
        )

    if (len(args) > 1 and args[1] == '--eval'):
        evaluate_model(big_lstm_model)
        exit()

    big_lstm_solver = CaptioningSolver(
    big_lstm_model, big_data,
        update_rule='adam',
        num_epochs=50,
        batch_size=128,
        optim_config={
        'learning_rate': 5e-4,
        },
        lr_decay=0.9965,
        verbose=True, print_every=10,
    )

    big_lstm_solver.train()
    with open('new_model.pkl', 'wb') as handle:
        pickle.dump(big_lstm_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot the training losses
    plt.plot(big_lstm_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    #plt.show()
    plt.savefig('result.png')

    evaluate_model(big_lstm_model)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        with open('tmp_model.pkl', 'wb') as handle:
            pickle.dump(big_lstm_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit(0)
