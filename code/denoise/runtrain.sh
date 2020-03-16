python train.py --net D --img_avg 1
python train.py --net D --img_avg 2
python train.py --net D --img_avg 4
python train.py --net D --img_avg 8
python train.py --net D --img_avg 16

python train.py --net M --img_avg 1
python train.py --net M --img_avg 2
python train.py --net M --img_avg 4
python train.py --net M --img_avg 8
python train.py --net M --img_avg 16

python train.py --net R --img_avg 1 --lr 0.0005 --batch_size 64
python train.py --net R --img_avg 2 --lr 0.0005 --batch_size 64
python train.py --net R --img_avg 4 --lr 0.0005 --batch_size 64
python train.py --net R --img_avg 8 --lr 0.0005 --batch_size 64
python train.py --net R --img_avg 16 --lr 0.0005 --batch_size 64