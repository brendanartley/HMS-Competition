import lightning.pytorch as pl

def get_logger(cfg, metrics):
    """
    Function to load wandb logger.
    """
    if cfg.no_wandb == True:
       return None, None

    # Get logger
    logger = pl.loggers.WandbLogger(
        project = cfg.project, 
        save_dir = None,
        )
    id_ = logger.experiment.id
    
    # Wandb metric summary options (min,max,mean,best,last,none): https://docs.wandb.ai/ref/python/run#define_metric
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_


def load_logger_and_callbacks(cfg, metrics):

    # Test Runs
    if cfg.fast_dev_run or cfg.overfit_batches > 0:
        return None, None

    # Other Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
    ]
    # Logger
    logger, id_ = get_logger(cfg, metrics)
    return logger, callbacks