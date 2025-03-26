source .venv/bin/activate
make -f Makefile save_folder=/home/woody/iwb3/iwb3021h/THESIS_RESULTS \
        submission=egenas \
        mode=T0 \
        augment=Basic \
        seed=4 \
        all

deactivate