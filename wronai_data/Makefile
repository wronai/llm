.PHONY: setup test collect clean

setup:
	bash setup_wronai_fixed.sh

test:
	source wronai_env/bin/activate && python scripts/test_data_collection.py

collect:
	source wronai_env/bin/activate && python scripts/collect_wronai_data_fixed.py

clean:
	rm -rf data/ test_data/ wronai_env/

help:
	@echo "WronAI Data Collection Commands:"
	@echo "  setup   - Setup environment"
	@echo "  test    - Run test collection (100MB)"
	@echo "  collect - Run full collection (5GB)"
	@echo "  clean   - Clean all data and env"
