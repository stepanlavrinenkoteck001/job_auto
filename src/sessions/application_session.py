from utils.logging_utils import create_logger
from pathlib import Path
from ingestion.database import VectorDataBase, PostgresDatabase
from preprocessing.spacy_prep import SpacyPrep
from modeling_clusterization.embedding import AbstractEmbedder
from modeling_clusterization.gpt_models import abstractGptModel
import pandas as pd


class ApplicationSession:
    """An application session to manage interactions between the user and the chatbot."""

    def __init__(self, postgres_db: PostgresDatabase, spc: SpacyPrep, embed_model: AbstractEmbedder,
                 qdrant_db: VectorDataBase, gpt_model: abstractGptModel,
                 fasttext_training_it_dataset_path: Path = None, log_level: str = 'INFO') -> None:
        """
        Initialize the ApplicationSession.

        Parameters:
            postgres_db (PostgresDatabase): The PostgreSQL database connection.
            spc (SpacyPrep): An instance of SpacyPrep for text preprocessing.
            embed_model (AbstractEmbedder): An embedding model for vectorizing sentences.
            qdrant_db (VectorDataBase): The Qdrant vector database connection.
            gpt_model (abstractGptModel): The GPT model for generating responses.
            fasttext_training_it_dataset_path (Path, optional): Path to the FastText training dataset (IT dataset). 
            log_level (str, optional): The log level for the logger (default is 'INFO').
        """
        self.logger = create_logger(log_level, log_name='application_session.py')
        self.postgres_db = postgres_db
        self.spc = spc
        self.embed_model = embed_model
        self.qdrant_db = qdrant_db
        self.gpt_model = gpt_model
        self.fasttext_training_it_dataset_path = fasttext_training_it_dataset_path

    def initialize_session(self):
        """
        Initialize the application session by connecting to the databases and loading the embedding model (if available).
        """
        self.postgres_db.connect()
        self.qdrant_db.connect()

        if self.embed_model.model_path.exists():
            self.embed_model.load()
        elif self.fasttext_training_it_dataset_path is not None and self.fasttext_training_it_dataset_path.exists():
            df_sentences = pd.read_csv(self.fasttext_training_it_dataset_path)
            sentences = df_sentences.values.tolist()
            self.embed_model.retrain(sentences)
            self.embed_model.save()

        self.logger.info('Application session initialized')
        return self

    def upsert_q_to_vec(self, user_id: str = 'b5f5f813-dafe-4cce-8f15-089bee4efacb'):
        """
        Vectorize the user's questions and upsert them into the Qdrant vector database.

        Parameters:
            user_id (str, optional): The user's ID (default is 'b5f5f813-dafe-4cce-8f15-089bee4efacb').
        """
        self.df_questions = self.postgres_db.query_questions_user(table_name_forms='forms_auto_fill',
                                                                  table_name_questions='questions',
                                                                  user_id=user_id)
        self.raw_q_list, self.raw_q_id_list = self.df_questions.name.tolist(), self.df_questions.question_id.tolist()
        self.clean_q_list = self.spc.prep_sentences_to_list_of_lists(self.raw_q_list)
        self.clean_qvec_list = self.embed_model.liststr_to_listvec(self.clean_q_list)

        self.qdrant_db.insert_rows(vectors=self.clean_qvec_list, id_list=self.raw_q_id_list, key=None, key_value=None)
        self.logger.info(f'Questions for user id {user_id} have been vectorized and added to the qdrant table')

        return self
    
    def substring_replacement(self, substring_to_replace:str, hist_question:str, hist_answer: str):
        """
        Replace historical questions and answers in a substring

        Parameters:
            substring_to_replace (str): A portion of the prompt input. It has hist_question and hist_answer, they will be replaced
            hist_question (str): Historical question, from all the job applications user has filled out
            hist_answer (str, optional): Historical answer matching the question

        Returns:
            str: The string with hist_question and hist_answer replaced
        """
        return (substring_to_replace.replace('hist_question', f'{hist_question}')
                            .replace('hist_answer', f'{hist_answer}'))
                            # .replace('new_question', f'{new_question}')

    def prompt_modification(self, append_string):
        """
        Modify gpt prompts to generate multiple different answers
        Parameters:
            append_string (str): String to append. Should contain a call to generate a difefrent answer


        Returns:
            self: Contains self.ret_list that should contain appended answers 
        """
        prompt = [{"role": "system", "content": self.system_string}, 
            {"role": "user", "content": self.user_string + append_string}]

        out = self.gpt_model.gpt_prompt_return(prompt)
        ret = str(out["choices"][0]["message"]['content'])
        
        if ret != '' or ret is not None:
            self.logger.info('GPT model generated a tuned answer')
            self.logger.info(f'Answer: {ret}')
            self.ret_list.append(ret)
            self.user_string = self.user_string + ' ' + ret
        else:
            self.logger.warning('GPT model could not generate a prediction. Returning an empty string')
            self.ret_list.append('')

        return self
    
    def query_tune_answer(self, user_string: str, system_string: str, substring_to_replace:str,
                          different_answer_user_string, new_question: str = 'Do you have experience in customer service?',
                          hist_qa_return_limit:int = 1, gpt_answer_return_limit: int = 3):
        """
        Generate a tuned answer using the GPT model based on user and system input.

        Parameters:
            user_string (str): The user's input string.
            system_string (str): The system's input string.
            substring_to_replace (str): Substring that will be inserted into teh rpompt. Has hist_answer and hist_question strings that will be replaced by historical qa pair
            different_answer_user_string (str): User string that prompts for a different answer
            new_question (str, optional): The new question to consider (default is 'Do you have experience in customer service?').
            hist_qa_return_hist_qa_return_limit (int): How many historical qa pairs to return
            gpt_answer_return_limit (int): How many different answer tuning should gpt return

        Returns:
            str: The generated tuned answer.
        """
        self.different_answer_user_string = different_answer_user_string
        self.hist_qa_return_limit = hist_qa_return_limit
        self.gpt_answer_return_limit = gpt_answer_return_limit

        self.clean_q = self.spc.prep_sentences_to_list_of_lists([new_question])
        self.clean_qvec = self.embed_model.liststr_to_listvec(self.clean_q)
        self.hist_question_id_score_dict = self.qdrant_db.query_app_q(query_vector=self.clean_qvec[0], limit=self.hist_qa_return_limit)


        
        self.hist_answer_question_dict = dict({self.postgres_db.query_answers_question_id(table_name_forms='forms_auto_fill', table_name_answers='answers', question_id= question_id)                                  
         for question_id in list(self.hist_question_id_score_dict.keys())})
        self.user_string = user_string
        self.substring_to_replace = substring_to_replace
        self.system_string = system_string
        self.new_question = new_question

        self.logger.info('Found and retrieved a matching historical question-answer pair(s)')
        self.logger.info(f'Question(s): {self.hist_answer_question_dict.values()}')
        self.logger.info(f'Answer(s): {self.hist_answer_question_dict.keys()}')
        self.logger.info(f'New question: {self.new_question}')
        
        self.substring_to_replace = ''.join(self.substring_replacement(self.substring_to_replace, hist_answer, hist_question) 
         for hist_question, hist_answer in self.hist_answer_question_dict.items())
        
        self.user_string = self.user_string.replace('substring_to_replace', self.substring_to_replace).replace('new_question', self.new_question)
        self.prompt = [{"role": "system", "content": self.system_string},
                       {"role": "user", "content": self.user_string}]
        
        self.out = self.gpt_model.gpt_prompt_return(self.prompt)
        self.ret = str(self.out["choices"][0]["message"]['content'])

        self.ret_list = []
        for _ in range(self.gpt_answer_return_limit):
            if gpt_answer_return_limit == 1:
                self.prompt_modification(append_string = '')
            else:
                self.prompt_modification(append_string = self.different_answer_user_string) 

        return self.hist_answer_question_dict, self.ret_list

