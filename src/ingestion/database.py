import os
from typing import List, Optional
from yaml import safe_load
from pathlib import Path
import numpy as np
import psycopg2
from psycopg2.sql import SQL, Identifier
import pandas.io.sql as psql
from dotenv import load_dotenv

load_dotenv()


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, \
    Filter, FieldCondition, MatchText, PointIdsList

from utils.logging_utils import create_logger
from exceptions.database_exceptions import DatabaseConnectionError

from abc import ABC, abstractclassmethod

class AbstractDatabase(ABC):
    
    @abstractclassmethod
    def connect():
        pass




class VectorDataBase(AbstractDatabase):
    """
    Represents a vector database.
    """

    def __init__(self, table_name: str, vec_size: int = 300, log_level:str = 'INFO'):
        """
        Initialize the VectorDataBase class.

        Args:
            table_name: Name of the table.
            table_schema: List of table schema.
            vec_size: Size of the vector.
            log_level: logger level, can be 'INFO', 'DEBUG', 'WARNING', 'ERROR'

        """
        self.table_name = table_name
        self.vec_size = vec_size
        self.client = None
        self.logger = create_logger(log_level, log_name = 'VectorDataBase')

    def reset_table(self, table_name: str, table_schema: List[str]):
        """
        Reset the table with the given table name and schema.
        This needs to be done in case you want to change the table you're working with currently.

        Args:
            table_name: Name of the table.
            table_schema: List of table schema.
        """
        self.table_name = table_name
        self.table_schema = table_schema
        self.logger.info(f'Qdrant table {self.table_name} has been reset')

    def connect(self):
        """
        Connect to the qdrant vector database.
        """
        try:
            self.client = QdrantClient(url=os.environ.get('QDRANT_DB_URL')+':'+os.environ.get('QDRANT_DB_PORT'))
            self.logger.info(f'Qdrant database has been connected')
        except:
            self.logger.error(f'Qdrant database not connected')


    def create_table(self):
        """
        Create a new table in the database.
        """
        self.client.recreate_collection(
            collection_name=self.table_name,
            vectors_config=VectorParams(size=self.vec_size, distance=Distance.COSINE),
        )
        self.logger.info(f'Qdrant table {self.table_name} has been created')

    
    def insert_rows(self, vectors: List[List[float]], id_list: List[int],
                    key: str, key_value:int):
        """
        Insert rows to the table with the given vectors and IDs.

        Args:
            vectors: List of vectors.
            id_list: List of IDs.
            key: String name of the key. For example: "user_id"
            key_value: String value of the key, For example, a key:key_value pair looks like this - "user_id": 123

        """
        status = self.client.get_collection(os.environ.get('QDRANT_QUESTIONS_TABLE_NAME'))
        if status.status[0] != 'g':
            self.logger.warning(f'Qdrant table {self.table_name} does not exist, creating a new table')
            self.create_table()
        if key_value == None:
            self.client.upsert(
            collection_name=self.table_name,
            points=[PointStruct(
                id=idx,
                vector=vector,
                
            )
            for vector, idx in zip(vectors, id_list)
            ]
        )
        else:
            self.client.upsert(
                collection_name=self.table_name,
                points=[PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                            key : key_value
                            }
                    
                )
                for vector, idx in zip(vectors, id_list)
                ]
            )
        self.logger.info(f'Qdrant table {self.table_name} rows have been inserted')


    def query_app_q(self, query_vector: List[float], key: str = None, key_value: str = None, limit: int = 1):
        """
        Perform a search query in the table with the given key, match text, query vector, and limit.

        Args:
            key_name: String name of the key. For example: "user_id"
            key_value: String value of the key, For example, a key_name:key_value pair looks like this - user_id: 123
            query_vector: Query vector.
            limit: Maximum number of search results to return.

        Returns:
            If we request only the top-matching question to be returned, then:
                self.question_id: id of the question
                self.score: cosine similarity score of the query (how good of a match is it)
            Otherwise:
                self.rows: A list of lists of question_id's and similarity scores with this format:  
                           [[id, score],[id, score],...]
        """
        status = self.client.get_collection(os.environ.get('QDRANT_QUESTIONS_TABLE_NAME'))
        if status.status[0] != 'g':
            self.logger.warning(f'Qdrant table {self.table_name} does not exist, creating a new table')
            self.create_table()

        if key_value is not None:
            self.rows = self.client.search(
                collection_name=self.table_name,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(
                        key=key,
                        match=MatchText(text=key_value),
                    )]
                ),
                limit=limit
            )
        else: 
            self.rows = self.client.search(
            collection_name=self.table_name,
            query_vector=query_vector,
            limit=limit
        )
        self.logger.info(f'Qdrant table {self.table_name} has been queried')
  
        self.hist_question_id_score_dict = {row.id: row.score for i, row in enumerate(self.rows)}
        return self.hist_question_id_score_dict
        
    def delete_rows(self, id_list: List[int]):
        """
        Delete rows from the table with the given IDs.

        Args:
            id_list: List of IDs.
        """
        status = self.client.get_collection(os.environ.get('QDRANT_QUESTIONS_TABLE_NAME'))
        if status.status[0] != 'g':
            self.logger.warning(f'Qdrant table {self.table_name} does not exist, creating a new table')
            self.logger.error('Cant delete rows!')

        self.client.delete(
            collection_name=self.table_name,
            points_selector=PointIdsList(
                points=id_list,
            ),
        )
        self.logger.info(f'Qdrant table {self.table_name} rows have been deleted')
    
    def update_vectors(self, vectors: List[List[float]], id_list: List[int]):
        """
        Update existing vectors with the new vectors, given their IDs.

        Args:
            vectors: List of vectors.
            id_list: List of IDs.

        """
        status = self.client.get_collection(os.environ.get('QDRANT_QUESTIONS_TABLE_NAME'))
        if status.status[0] != 'g':
            self.logger.warning(f'Qdrant table {self.table_name} does not exist, creating a new table')
            self.create_table()

        self.client.update_vectors(
            collection_name=self.table_name,
            points=[PointStruct(
                id=idx,
                vector=vector,
                
            )
            for vector, idx in zip(vectors, id_list)
            ]
        )
        self.logger.info(f'Qdrant table {self.table_name} vectors have been updated')






class PostgresDatabase(AbstractDatabase):
    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize the PostgresDatabase class.
        ------------
        Database structure description:
        1) таблица forms_auto_fill являеться связывающей вопросы и ответы. 

        
        table forms_auto_fill:

        column: question_id foreign key of table questions where column: name contains information about question

        column: answer_id foreign key of table answers where column: name contains information about answer

        2) таблица posting являеться таблицей хранящей информацию о job полученной от SerpApi либо по средствам GoogleExtention. 

        table posting:

        column: description, contains information about job description

        column: title, contains information about name of job
        ----------------

        Args:
            log_level: logger level, can be 'INFO', 'DEBUG', 'WARNING', 'ERROR'
        """
        self.logger = create_logger(log_level, log_name='PostgresDatabase')

    def connect(self):
        """
        Connects to the PostgreSQL database.

        Returns:
            self
        """
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_DB_URL'),
                port=os.getenv('POSTGRES_DB_PORT'),
                dbname=os.getenv('POSTGRES_DB_NAME'),
                user=os.getenv('POSTGRES_DB_USER'),
                password=os.getenv('POSTGRES_DB_PASS')
            )
            self.cursor = self.conn.cursor()
            self.logger.info(f"Postgres database {os.getenv('POSTGRES_DB_NAME_STAGE')} has been connected ")
        except DatabaseConnectionError as err:
            raise DatabaseConnectionError(f"Connection error {err} or connection parameters have not been set")

        return self

    def execute_query(self, query: str, fetchone: bool = True):
        """
        Executes the provided SQL query.

        Args:
            query: SQL query to execute.
            fetchone: Whether to fetch only one row or all rows. Default is True.

        Returns:
            self
        """
        self.cursor.execute(query)
        if fetchone:
            self.logger.info("Selecting rows using cursor.fetchone")
            self.row = self.cursor.fetchone()
        else:
            self.logger.info("Selecting rows using cursor.fetchall")
            self.row = self.cursor.fetchall()
        return self

    def display_results(self):
        """
        Displays the results of the executed query.
        """

        while self.row is not None:
            self.logger.info(self.row)
            self.row = self.cursor.fetchone()
        self.logger.info("Print each row and its column values")

    def print_schema(self, table_name: str):
        """
        Prints the schema of the specified table.

        Args:
            table_name: Name of the table.

        """

        query = f"""SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}';"""

        # query =  SQL(f"""SELECT column_name, data_type, is_nullable
        # FROM information_schema.columns
        # WHERE table_name = '{ Identifier(table_name)}';""")

        self.logger.info('Printing schema: column name, type, is nullable')
        self.execute_query(query).display_results()

    def query_questions_user(self, table_name_forms: str = 'forms_auto_fill',
                             table_name_questions: str = 'questions',
                             user_id: str = 'b5f5f813-dafe-4cce-8f15-089bee4efacb'):
        """
        Gets all the question ids and question texts for one user by performing a join between the forms table (user profile)
        and the questions table. Meant to be used to update the vector database with new question vectors.

        Args:
            table_name_forms: Name of the forms table. Default is 'forms_auto_fill'.
            table_name_questions: Name of the questions table. Default is 'questions'.
            user_id: ID of the user. Default is 'b5f5f813-dafe-4cce-8f15-089bee4efacb'.

        Returns:
            self.df: Pandas dataframe with two columns: question_id and name. They contain an id of teh question and question text, respectively
        """

        query = f"""SELECT 
        question_id, 
        name
        FROM {table_name_forms} f
        LEFT JOIN {table_name_questions} q ON q.id = f.question_id
        WHERE f.user_id = '{user_id}'
        """

        # query = SQL(f"""SELECT 
        # question_id, 
        # name
        # FROM { Identifier(table_name_forms)} f
        # LEFT JOIN { Identifier(table_name_questions)} q ON q.id = f.question_id
        # WHERE f.user_id = '{ Identifier(user_id)}'
        # """)

        self.df = psql.read_sql(query, self.conn)
        if self.df.shape[0] < 1:
            self.logger.warning(f'No questions for this user {user_id} were found')
        else:
            self.logger.info('fetched question ids and text')

        return self.df


    def query_answers_question_id(self, table_name_forms: str = 'forms_auto_fill',
                             table_name_answers: str = 'answers',
                             table_name_questions: str = 'questions',
                             question_id: str = '662c8e7c-7ed8-4371-858d-58efa255ddec'):
        """
        Given a question_id, find the matching answer text. Also retrive the historical question text(saved in db)
        Meant to be used in combination with vector database, when trying to find an existing question-answer pair closest to a scraped question
        Then using gpt to adjust historical question-answer pair + new question to create a new, adjusted answer
        Args:
            table_name_forms: Name of the forms table. Default is 'forms_auto_fill'.
            table_name_answers: Name of the answers table. Default is 'answers'.
            table_name_questions: Name of the questions table. Default is 'questions'.
            question_id: ID of the user. Default is '662c8e7c-7ed8-4371-858d-58efa255ddec'.

        Returns:
            self.answer: text of the answer for the scraped (given) question_id
            self.question: text of the historical question (saved in db)

        """


        query = f"""SELECT 
        *
        FROM {table_name_forms} f
        INNER JOIN {table_name_answers} a ON a.id = f.answer_id
        INNER JOIN {table_name_questions} q ON q.id = f.question_id
        WHERE f.question_id = '{question_id}'

        """
        # query =  SQL(f"""SELECT 
        # *
        # FROM { Identifier(table_name_forms)} f
        # INNER JOIN { Identifier(table_name_answers)} a ON a.id = f.answer_id
        # INNER JOIN { Identifier(table_name_questions)} q ON q.id = f.question_id
        # WHERE f.question_id = '{ Identifier(question_id)}'
        # """)
        self.df_a = psql.read_sql(query, self.conn)
        if self.df_a.shape[0] < 1:
            self.logger.warning(f'No answers were found for this question {question_id}')
        else:
            self.logger.info('fetched the text of the answer')

        self.answer, self.question = self.df_a.name.iloc[0,0], self.df_a.name.iloc[0,1]
        return self.answer, self.question

   

