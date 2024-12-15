lint:
	docker-compose up lint --abort-on-container-exit

lint-only-error:
	pylint 'src/**/*.py' --fail-under=9.0 --fail-on=E --disable=W,R,C

ruff-lint:
	uv run ruff check

ruff-lint-fix:
	uv run ruff check --fix

test:
	docker-compose up test --abort-on-container-exit

docker-dev:
	docker-compose up api --build

dev:
	uvicorn main:app --port 3100 --reload

install:
	pip install --no-cache-dir -r src/app/requirements.txt

install-dev:
	pip install --no-cache-dir -r src/app/requirements.dev.txt

migration-create:
	cd src/app && alembic revision -m "$(name)"

migration-up:
	cd src/app && alembic upgrade heads

migration-history:
	cd src/app && alembic history

migration-merge:
	cd src/app && alembic merge heads

check-types:
	pyright

pre-commit:
	pre-commit run --all-files
