
from utils.logging_utils import create_logger
from yaml import safe_load
from pathlib import Path
from ingestion.database import VectorDataBase, PostgresDatabase
from utils.logging_utils import create_logger
from preprocessing.spacy_prep import SpacyPrep
from modeling_clusterization.embedding import FastTextModel
from modeling_clusterization.gpt_models import ChatGpt, Gpt4All
from sessions.application_session import ApplicationSession
import os 

if __name__ == "__main__":
        
    
    with open('src/universal_config.yaml', 'r') as file:
    # with open('universal_config.yaml', 'r') as file:

        cf = safe_load(file)

    cf_db_p = cf['database']['postgres']
    cf_db_q = cf['database']['qdrant']


    with open('src/application_questions_config.yaml', 'r') as file:
    # with open('application_questions_config.yaml', 'r') as file:

        cf_appq = safe_load(file)

    # general paths
    parent_folder_path = Path(os.getenv('PARENT_FOLDER_PATH'))
    data_path = parent_folder_path / 'data' 
    app_questions_data_path = data_path / 'app_questions' 
    embed_model_path = app_questions_data_path / 'artifacts' / 'fasttext.model'
    fasttext_training_it_dataset_path = app_questions_data_path / 'prep_data' / 'prep_sentences_50K.csv'

    # paths to gpt binaries and generated output csv files
    cf_gpt_model_to_use = os.getenv('GPT_MODEL_TO_USE')  # chatgpt or gpt4all
    model_path = parent_folder_path / 'data' / cf['gpt_model_constants']['model_path']
    gpt_model_str =  cf['gpt_model_constants'][cf_gpt_model_to_use]['gpt_model_str'] #or use gpt4all constants

    # strings to generate prompts
    system_string = cf_appq['gpt_model_constants']['default_prompt_strings']['system_string']
    user_string = cf_appq['gpt_model_constants']['default_prompt_strings']['user_string']
    substring_to_replace = cf_appq['gpt_model_constants']['default_prompt_strings']['substring_to_replace']
    different_answer_user_string = cf_appq['gpt_model_constants']['default_prompt_strings']['different_answer_user_string']

    # other gpt constants  - use .env variables, so that these can be altered easily
    hist_qa_return_limit = int(os.getenv('HIST_QA_RETURN_LIMIT'))
    gpt_answer_return_limit = int(os.getenv('GPT_ANSWER_RETURN_LIMIT'))


    # logging constants and pipeline logger
    log_level = cf['utils']['log_level']
    tpb_logger = create_logger(log_level, log_name = 'application_questions_pipeline_log')
    tpb_logger.info('running application_questions_pipeline.py')

    postgres_db = PostgresDatabase()
    spc = SpacyPrep()
    embed_model = FastTextModel(embed_model_path)
    qdrant_db = VectorDataBase(os.getenv('QDRANT_QUESTIONS_TABLE_NAME'),
                                    cf_db_q['vec_size'])

    if cf_gpt_model_to_use == 'chatgpt':
        chatgpt_api_timing_delay =  cf_appq['gpt_model_constants'][cf_gpt_model_to_use]['chatgpt_api_timing_delay']
        gpt_model = ChatGpt(gpt_model_str, model_path, chatgpt_api_timing_delay, log_level)
    else:
        gpt_model = Gpt4All(gpt_model_str, model_path, log_level)

    app_session = ApplicationSession(postgres_db, spc, 
                    embed_model, qdrant_db,
                    gpt_model, fasttext_training_it_dataset_path,
                    log_level)
    app_session.initialize_session()
    # functionality #1 - for a given user, convert new text questions into vectors and insert into qdrant
    app_session.upsert_q_to_vec(user_id = 'b5f5f813-dafe-4cce-8f15-089bee4efacb')
    
    # functionality #2 - for a given user and a new application question, find most similar historical qa pair
    #  and tune them via gpt to produce a new answer
    hist_answer_question_dict, ret_list = app_session.query_tune_answer(user_string, system_string, substring_to_replace, 
                            different_answer_user_string, new_question = 'Do you have experience in customer service?',
                            hist_qa_return_limit=hist_qa_return_limit, gpt_answer_return_limit = gpt_answer_return_limit
                            )

