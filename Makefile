## ----------------------------------------------------------------------
## This Makefile includes simplified commands for development
## ----------------------------------------------------------------------

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

format: ## Formats the code, import statements and type hints
	black .
	isort .
	mypy .

format: ## Runs the tests in the tests directory
	pytest tests/ 

train: ## Runs a training session following the configuration in config.ini
	poetry run pong_dqn_rl/training.py
