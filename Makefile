OS := $(shell uname -s | tr A-Z a-z)

# OS 확인 후, 패키지 관리자 명령어 지정
ifeq ($(OS), darwin) # MacOS인 경우
	DEPENDENCY_MGMT = brew
else ifeq ($(OS), linux) # Linux인 경우
	DEPENDENCY_MGMT = apt-get
endif

# Poetry 존재 여부 확인 후, 설치명령어를 변수에 넣음
ifeq (,$(shell which poetry))
	POETRY_CMD = curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
else
	POETRY_CMD = poetry version
endif

# pyenv 존재 여부 확인 후, 설치명령어를 변수에 넣음
ifeq (,$(shell which pyenv))
	PYENV_CMD = $(install_dep) install pyenv, pyenv-virtualenv
else
	PYENV_CMD = pyenv version
endif

init: install-python-package set-commit-template set-pre-commit
format: pre-commit-check
clean: clean-pyc clean-test

####  settings  ####
install-dependency-package:
	$(DEPENDENCY_MGMT) update

install-pyenv:
	$(PYENV_CMD)
	export PYENV_ROOT="$HOME/.pyenv"
	export PATH="$PYENV_ROOT/bin:$PATH"
	export PATH="$PYENV_ROOT/shims:$PATH"
	eval "$(pyenv init --path)"
	eval "$(pyenv init -)"
	eval "$(pyenv virtualenv-init -)"

install-poetry:
	$(POETRY_CMD)

install-python-package:
	poetry install

set-commit-template:
	git config --local commit.template .gitmessage.txt

set-pre-commit:
	poetry run pre-commit install

unset-precommit:
	poetry run pre-commit uninstall

####  format  ####
pre-commit-check:
	poetry run pre-commit run --all-files

####  test  ####
pytest:
	pytest -o log_cli=true --disable-pytest-warnings --cov-report term-missing tests/

####  clean  ####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf tests/output
	rm -rf *.log
	rm -rf dist/
#################
