python train.py --net ours --lr_h5 ../../net_data/avg1_64_32.h5 --ngpu 3

python train.py --net RCAN --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2
python train.py --net RDN --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2
python train.py --net SAN --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2
python train.py --net SRFBN --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2
python train.py --net ESRGAN --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2
python train.py --net EPSR --lr_h5 ../../net_data/avg400_64_32.h5 --ngpu 2