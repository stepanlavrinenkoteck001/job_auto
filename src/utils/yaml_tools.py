import yaml

def read_yaml_files(*paths):
    contents = []
    for path in paths:
        with open(path, "r") as file:
            content = yaml.load(file, Loader=yaml.FullLoader)
            contents.append(content)
    return contents