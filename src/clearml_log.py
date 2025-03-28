def clearml_logging(config, logger):
    for hyperparam in config:
        if hyperparam[0] in ['data_config', 'mlflow_config']:
            for hyperparam_data in hyperparam[1]:
                report(hyperparam_data, logger)
        else:
            report(hyperparam, logger)


def report(hyperparam, logger):
    if type(hyperparam[1]) in [str, dict, list]:
        logger.report_text(f'{hyperparam[0]}: {hyperparam[1]}')
    else:
        logger.report_single_value(name=hyperparam[0], value=hyperparam[1])