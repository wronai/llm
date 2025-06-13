import yaml

def check_config():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Learning rate type: {type(config['training']['learning_rate'])}")
    print(f"Learning rate value: {config['training']['learning_rate']}")

if __name__ == "__main__":
    check_config()
