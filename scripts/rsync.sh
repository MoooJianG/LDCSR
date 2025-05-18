# from autodl to cp3090
rsync -avz -e 'ssh -p 37035' root@connect.bjb1.seetacloud.com:/root/autodl-tmp/LatentDiffusionSR/logs/* /root/LatentDiffusionSR/logs/

# from cp3090 to bfsu-dlcoud
sudo rsync -avz -e 'ssh -p 30001' ./logs/* root@bfsu-dlcloud:/workspace/workspace/cp3090/LatentDiffsuionSR-logs/


sudo rsync -avz -e 'ssh -p 26135' ./logs/autoencoder_ms_vq_v8_aid/2024-04-19T22-28-21/checkpoints/epoch=0046-metric=0.21.ckpt root@connect.bjb1.seetacloud.com://root/autodl-tmp/LatentDiffusionSR/logs/autoencoder_ms_vq_v8_aid/2024-04-19T22-28-21/checkpoints/


rsync -avz -e 'ssh -p 34019' /root/LatentDiffusionSR/load/AID root@connect.bjb1.seetacloud.com:/root/autodl-tmp/LatentDiffusionSR/logs/* 