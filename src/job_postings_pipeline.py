from utils.logging_utils import create_logger
from yaml import safe_load
from pathlib import Path
import os

from ingestion.jp_ingestion import jobPostIingest
from modeling_clusterization.gpt_models import ChatGpt, gptPredict, Gpt4All  # can also import Gpt4All class to use with free model


if __name__ == "__main__":

    ## constants from config
    with open('src/universal_config.yaml', 'r') as file:
        cf = safe_load(file)

    with open('src/job_postings_config.yaml', 'r') as file:
        cf_jobp = safe_load(file)

    parent_folder_path = Path(os.getenv('PARENT_FOLDER_PATH'))

    # paths to raw data and preprocessed data
    job_descr_path = parent_folder_path / 'data' / cf_jobp['ingestion']['data_paths']['job_descr_path']
    prep_data_path = parent_folder_path / 'data' / cf_jobp['ingestion']['data_paths']['prep_data_path']

    # paths to gpt binaries and generated output csv files
    cf_gpt_model_to_use = cf_jobp['gpt_model_constants']['cf_gpt_model_to_use']
    model_path = parent_folder_path / 'data' / cf['gpt_model_constants']['model_path']
    gpt_output_path = prep_data_path / cf['gpt_model_constants'][cf_gpt_model_to_use]['gpt_output_path']  #or use gpt4all constants
    gpt_model_str =  cf['gpt_model_constants'][cf_gpt_model_to_use]['gpt_model_str'] #or use gpt4all constants
    if cf_gpt_model_to_use == 'chatgpt':
        chatgpt_api_timing_delay =  cf['gpt_model_constants'][cf_gpt_model_to_use]['chatgpt_api_timing_delay']

    # strings to generate prompts
    system_string = cf_jobp['gpt_model_constants']['default_prompt_strings']['system_string']
    user_string = cf_jobp['gpt_model_constants']['default_prompt_strings']['user_string']

    # logging constants and pipeline logger
    log_level = cf['utils']['log_level']
    tpa_logger = create_logger(log_level, log_name = 'training_pipeline_a_log')

    tpa_logger.info('running training_pipeline_a.py')
    df = jobPostIingest(job_descr_path, gpt_output_path, log_level).ingest_jp()
    
    if cf_gpt_model_to_use == 'chatgpt':
        chatgpt_api_timing_delay =  cf_jobp['gpt_model_constants'][cf_gpt_model_to_use]['chatgpt_api_timing_delay']
        gpt_model = ChatGpt(gpt_model_str, model_path, chatgpt_api_timing_delay, log_level)
    else:
        gpt_model = Gpt4All(gpt_model_str, model_path, log_level)

    gpt = gptPredict(df, gpt_model, system_string, user_string, gpt_output_path, log_level)
    gpt.apply_lambda_save_csv()
