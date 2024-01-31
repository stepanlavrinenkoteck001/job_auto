import logging
loggers = {} 
def create_logger(
    log_level:str ='INFO', 
    log_name:str = 'logfile',
    export_log: bool = False, # unable to write logs in Databricks
    save_dir:str = ''):
    
    if log_name in loggers.keys():
        logger = loggers.get(log_name)
    else:
        # create logger
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        handler1 = logging.StreamHandler()
        handler1.setLevel(log_level)
        pathname = log_name
        if len(save_dir)>0:
            pathname = f'{save_dir}{pathname}'
        # create formatter 
        formatter= logging.Formatter('[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
        
        # add formatter to ch
        handler1.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(handler1)
        if export_log:
            try:
                handler2 = logging.FileHandler(filename=f'{pathname}.log', mode='a')
            except:
                open(f'{pathname}.log', mode= 'x')
                handler2= logging.FileHandler(filename=f'{pathname}.log', mode='w')
            handler2.setLevel(log_level)
            handler2.setFormatter(formatter)
            logger.addHandler(handler2)
            
        loggers[log_name] = logger
    
    return logger