from pathlib import Path
import pandas as pd
import json
import numpy as np
import time

from utils.logging_utils import create_logger

from typing import List
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_vectors

from abc import ABC, abstractclassmethod

class AbstractEmbedder(ABC):
    
    @abstractclassmethod
    def load():
        pass

    @abstractclassmethod
    def save():
        pass

    @abstractclassmethod
    def retrain():
        pass
    
    @abstractclassmethod
    def liststr_to_listvec():
        pass



class FastTextModel(AbstractEmbedder):
    """
    A wrapper class for Gensim's FastText model.

    Attributes:
        model_path (str): The path to the saved model file.
        model: The loaded FastText model.

    Methods:
        load(): Loads a saved model from disk or initializes a new model if no path is given.
        save(): Saves the current model to disk.
        retrain(sentences: List[str]): Retrains the model on the given list of sentences.
        str_to_vec(sentences: List[str]) -> List[float]: Converts a list of sentences to a list of sentence vectors.
        str_to_listvec(sentences: List[List[str]]) -> List[List[float]]: Converts a list of lists of sentences to a 
            list of lists of sentence vectors.
    """

    def __init__(self, model_path: str, epochs:int = 5, min_count:int = 5, log_level: str = 'INFO'):
        """
        Initializes an instance of FastTextModel based on saved path.

        Args:
            model_path (str): The path to the saved model file. If None, will create a sample fasttext model based on common texts
            epochs (int): Number of epochs to train the model for. More epochs = better result, but will consume more time. Performance will flatten out after too many epochs
            min_count (int): Number of repeats of a word that is needed in order to add it to vocabulary. Higher number = less memory, but less frequent words will be skipped
        """
        self.model_path = model_path
        self.epochs = epochs
        self.model = None
        self.min_count = min_count
        self.logger = create_logger(log_level, log_name='FastTextModel')


    def retrain(self, sentences: List[str]):
        """
        Retrains the model on the given list of sentences.

        Args:
            sentences (List[str]): A list of sentences to train the model on.
        """

        if self.model is None:
            self.model = FastText(vector_size=300, window=3, min_count=self.min_count)  # instantiate
            self.model.build_vocab(sentences)
        else:
            self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=len(sentences), epochs=self.epochs)
        self.logger.info('Retraining fasttext model. Dont forget to run FastTextmodel.save() after')
        return self
    
    def load(self):
        """
        Loads a saved model from disk 
        """
        if self.model_path.exists:
            fname = get_tmpfile(self.model_path)
            self.model = FastText.load(fname)
            self.logger.info('Loaded fast text model from disk')
            return self
        else:
            self.logger.error('No model found on the hard-drive. Check that the model file fasttext.model exists or use FastTextModel.retrain() to retrain the model on a corpus of words')
        
        
    def load_keyed_vectors(self):
        """
        Loads only keyed vectors. This consumes less cpu than full load during inference. 
        Note that you can't retrain in this case
        """
        fname = datapath(self.model_path)
        self.model = load_facebook_vectors(fname)
        self.logger.info('Loaded keyed vectors only to speed up the str->vec conversion. Note that you cant retrain in this state')
        return self

    def save(self):
        """
        Saves the current model to disk.
        """
        fname = get_tmpfile(self.model_path)
        self.model.save(fname)
        self.logger.info('Saved model to drive')
        return self


    def liststr_to_listvec(self, sentences: List[str]) -> List[float]:
        """
        Converts a list of sentences to a list of sentence vectors.

        Args:
            sentences (List[str]): A list of sentences to convert.

        Returns:
            List[float]: A list of sentence vectors.
        """
        if self.model is None:
            self.load()
        self.logger.info('Converting strings of text to vectors via fasttext model')
        return [self.model.wv.get_sentence_vector(sent).tolist() for sent in sentences]

