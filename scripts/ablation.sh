CUDA_VISIBLE_DEVICES=0 nohup python train.py --config configs/second_stage_van_v4_vq.yaml > nohup.gpu0.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config configs/first_stage_kl_v6_a3.yaml > nohup.gpu1.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --config configs/first_stage_kl_v6_a4.yaml > nohup.gpu2.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --config configs/first_stage_kl_v6_a1.yaml > nohup.gpu3.out &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config configs/second_stage_van_v4_wokd.yaml > nohup.gpu0kd.out &