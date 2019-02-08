# DD
python -m train --datadir=data --bmname=DD --cuda=0 --max-nodes=500 --epochs=1000

# ENZYMES
python -m train --datadir=data --bmname=ENZYMES --cuda=3 --max-nodes=100 --num-classes=6

