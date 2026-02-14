venv:
	python3 -m venv venv

activate:
	source venv/bin/activate

deactivate:
	deactivate
	
install:
	pip install -r requirements.txt