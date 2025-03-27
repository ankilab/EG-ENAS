#Low cost EG-ENAS
source .venv/bin/activate
# Basic: RandomErasing+ RandomCrop + HorizontalFlip
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0+ \
        augment=Basic \
        seed=1 \
        all

deactivate