import string
import spacy 
from typing import List
from pathlib import Path
import pandas as pd
import json
import numpy as np
import time
import os
from utils.logging_utils import create_logger




class SpacyPrep:
    """
    A class to clean a string of text using spaCy NLP library.
    """

    def __init__(self, log_level: str = 'INFO'):
        """
        Initializes the SpacyPrep object.

        Parameters:
            nlp (spacy.lang): The spaCy language model to use for text processing.
        """
        # load spacy dict in case it was not installed automatically
        os.system("python -m spacy download en_core_web_sm")

        self.nlp = spacy.load("en_core_web_sm")
        self.excpetion_list = ['\n']
        self.logger = create_logger(log_level, log_name='SpacyPrep')

    def _remove_punct(self, doc):
        """
        Internal method to remove punctuation tokens from a spaCy document.

        Parameters:
             doc (spacy.tokens.Doc): The input spaCy document.

        Returns:
            generator: Yields spaCy tokens that are not punctuation.
        """

        return (t for t in doc if t.text not in string.punctuation)

    def _remove_stop_words(self, doc):
        """
        Internal method to remove stop words from a spaCy document.

        Parameters:
            doc (spacy.tokens.Doc): The input spaCy document.

        Returns:
            generator: Yields spaCy tokens that are not stop words.
        """

        return (t for t in doc if not t.is_stop in self.excpetion_list)

    def _lemmatize(self, doc):
        """
        Internal method to lemmatize a spaCy document.

        Parameters:
            doc (spacy.tokens.Doc): The input spaCy document.

        Returns:
            list of str: List of lemmatized tokens.
        """

        return [t.lemma_ for t in doc]

    def prep_words_to_list(self, text:str) -> List[str]:
        """
        Pre-process input text into a list of clean lemmatized tokens.

        Parameters:
            text (str): The input text to pre-process.

        Returns:
            list of str: List of clean lemmatized tokens.
        """
        doc = self.nlp(text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        self.logger.info('cleaning words to a list words')
        return self._lemmatize(removed_stop_words)

    def prep_sentences_to_list_of_lists(self, text: List[str]) -> List[List[str]]:
        """
        Pre-process input text into a list of lists of clean lemmatized tokens representing sentences.

        Parameters:
            text (List of strings/sentences): The input sentences to pre-process.

        Returns:
            list of list of str: List of lists of clean lemmatized tokens representing sentences.
        """

        doclist = list(self.nlp.pipe(text))
        lemm_docs = []
        for doc in doclist:
            lemm_docs.append(self.prep_words_to_list(doc))
        self.logger.info('cleaning list of sentences to a list of lists of words')
        return lemm_docs