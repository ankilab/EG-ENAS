#Low cost EG-ENAS
# fisher + jacob_cov zero-cost proxies based augmentation selection 

if [ ! -d "datasets" ]; then
    printf "❌ Datasets folder has not been created. Exiting...\n"
    exit 1  # Stop the script
fi

source .venv/bin/activate
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0 \
        augment=Proxy \
        seed=1 \
        all

deactivate