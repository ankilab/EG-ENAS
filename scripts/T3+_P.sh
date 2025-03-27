#Evolutionary NAS with Random Forest based population initialization 
#fisher + jacob_cov zero-cost proxies based augmentation selection 
source .venv/bin/activate

if [ ! -d "datasets" ]; then
    printf "❌ Datasets folder has not been created. Exiting...\n"
    exit 1  # Stop the script
fi

make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T3+ \
        augment=Proxy \
        seed=1 \
        all

deactivate