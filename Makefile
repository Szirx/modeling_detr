install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python3 src/train.py configs/config.yaml