import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    for word_pos in range(0, len(test_set.get_all_Xlengths())):
        sequences, sequences_length = test_set.get_item_Xlengths(word_pos)
        word_logL = {}

        for word in models:
            model = models[word]

            try:
                word_logL[word] = model.score(sequences, sequences_length)
            except:
                # if not found the likelyhood is set to -inf
                word_logL[word] = float("-inf")
                continue

        probabilities.append(word_logL)
        guesses.append(max(word_logL, key=word_logL.get))

    return probabilities, guesses
