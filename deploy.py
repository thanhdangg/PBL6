env_list= []

with open(".env", "r") as fh_env:
    for env_line in fh_env:
        env_list.append(env_line.replace('\n', ''))

with open(".env-sample", "a") as fh_env_example:
    for variable in env_list:
        var_only = variable.split("=")[0]
        fh_env_example.write(f"{var_only}=XXXXX\n")