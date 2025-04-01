install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python3 src/train.py configs/config.yaml

train_ckpt:
	PYTHONPATH=. python3 src/train_ckpt.py configs/config.yaml

tmux:
	tmux attach -t detr