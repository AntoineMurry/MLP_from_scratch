check_code:
	@flake8 MLP_from_scratch/*.py

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__
	@rm -fr __pycache__
	@rm -fr build
	@rm -fr dist
	@rm -fr MLP_from_scratch-*.dist-info
	@rm -fr MLP_from_scratch.egg-info

all: clean install test check_code

install: clean wheel
	@pip3 install -U dist/*.whl

prod_install: wheel
	@ansible-playbook -i ansible/all.serverlist \
	                                        --extra-vars "venv=venv" \
	                                        --extra-vars "user=fox" \
	                                        ansible/playbook_deploy.yml
	@echo Package installed on host remote host.


install_requirements:
	@pip3 install -r requirements.txt

wheel:
	@rm -f dist/*.whl
	@python3 setup.py bdist_wheel  # --universal if you are python2&3

test:
	@coverage run -m unittest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*,MLP_from_scratch/*

