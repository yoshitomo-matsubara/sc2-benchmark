def overwrite_config(org_config, sub_config):
    """
    Overwrites a configuration (dict).

    :param org_config: A (nested) dictionary of configuration to be updated.
    :type org_config: dict
    :param sub_config: A (nested) dictionary to be added to org_config.
    :type sub_config: dict
    """
    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = sub_value
        else:
            org_config[sub_key] = sub_value
