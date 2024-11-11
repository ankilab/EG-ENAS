source /home/hpc/iwb3/iwb3021h/NAS_CHALLENGE/NAS_Challenge_AutoML_2024/.testvenv/bin/activate
cd /home/hpc/iwb3/iwb3021h/NAS_CHALLENGE/NAS_Challenge_AutoML_2024
make -f Makefile save_folder=/home/woody/iwb3/iwb3021h/THESIS_RESULTS \
        submission=egenas \
        mode=T0 \
        augment=Basic \
        seed=4 \
        all

deactivate