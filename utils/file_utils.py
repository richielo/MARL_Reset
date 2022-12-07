import yaml

def parse_yaml(file_path):
    with open(file_path, "r") as stream: 
        try:
            yaml_output = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_output