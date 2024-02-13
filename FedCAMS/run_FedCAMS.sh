source /home/hans/WorkSpace/venv/bin/activate
#fedadam fedadagrad fedyogi fedams fedcams
optimizer="fedadam"
model="resnet18"
dataset="cifar10"
gpu=0
iid=0
if [ $optimizer != "fedcams" ]; then
  if [ $iid == 0 ]; then
    nohup python-hans -u federated_main.py -m=$model -d=$dataset -g=$gpu -o=$optimizer --iid=$iid > $optimizer-$model-$dataset-noniid.txt 2>&1 &
  else
    nohup python-hans -u federated_main.py -m=$model -d=$dataset -g=$gpu -o=$optimizer --iid=$iid > $optimizer-$model-$dataset-iid.txt 2>&1 &
  fi
else
  if [ $iid == 0 ]; then
    nohup python-hans -u federated_main-ef.py -m=$model -d=$dataset -g=$gpu -o="fedams" --iid=$iid > $optimizer-$model-$dataset-noniid.txt 2>&1 &
  else
    nohup python-hans -u federated_main-ef.py -m=$model -d=$dataset -g=$gpu -o="fedams" --iid=$iid > $optimizer-$model-$dataset-iid.txt 2>&1 &
  fi
fi
