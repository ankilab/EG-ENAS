source .venv/bin/activate
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0+ \
        augment=Proxy \
        seed=4 \
        all

deactivate