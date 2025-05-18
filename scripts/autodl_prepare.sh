ln -s ../../autodl-fs/hlwu/LatentDiffSR-logs ./logs
unzip ../AID_dataset.zip
mkdir load
cp -r ../AID ./load/
pip install natsort