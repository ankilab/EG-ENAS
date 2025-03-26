source .venv/bin/activate
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0+ \
        augment=Proxy \
        seed=4 \
        pretrained_pool_path=/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/pretrained_pool/ \
        all

deactivate