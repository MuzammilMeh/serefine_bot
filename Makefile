.PHONY: check
check: check-format lint

.PHONY: check-format
check-format: ## Dry-run code formatter
	poetry run black ./app --check
	poetry run isort ./app --profile black --check

.PHONY: lint
lint: ## Run linter
	poetry run pylint ./app 
 
.PHONY: format
format: ## Run code formatter
	poetry run black ./app
	poetry run isort ./app --profile black

.PHONY: run
run: ## Run the application
	poetry run uvicorn main:app --reload