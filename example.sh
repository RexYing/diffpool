# ENZYMES
python -m train --datadir=data --bmname=ENZYMES --cuda=3 --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=1 --num-classes=6 --method=soft-assign

# DD
python -m train --datadir=data --bmname=DD --cuda=0 --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=1 --num-classes=2 --method=soft-assign


python -m train --bmname=REDDIT-MULTI-12K --num-classes=11 --hidden-dim=64 --output-dim=64 --method=soft-assign --num-pool=2

python -m train --bmname=COLLAB --num-classes=3 --hidden-dim=32 --output-dim=32 --method=soft-assign --num-pool=2

python -m train --bmname=PROTEINS --num-classes=2 --hidden-dim=64 --output-dim=64 --method=soft-assign
