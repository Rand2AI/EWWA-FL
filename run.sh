source /home/hans/WorkSpace/venv/bin/activate
method='EWWA'
optimizer='adam'
network='resnet'
layer=34
dataset='cifar10'
gpu=1
iid=0
if [ $iid == 0 ]; then
  nohup python -u Train.py -m=$method -o=$optimizer -n=$network -l=$layer -d=$dataset -g=$gpu --iid=$iid > $method-$optimizer-$network$layer-$dataset-noniid.txt 2>&1 &
else
  nohup python -u Train.py -m=$method -o=$optimizer -n=$network -l=$layer -d=$dataset -g=$gpu --iid=$iid > $method-$optimizer-$network$layer-$dataset-iid.txt 2>&1 &
fi
