#full EG-ENAS: Evolutionary NAS + Regressor based population initialization + Weights transfer
source .venv/bin/activate

if [ ! -d "datasets" ]; then
    printf "❌ Datasets folder has not been created. Exiting...\n"
    exit 1  # Stop the script
fi

if [ ! -d "pretrained_pool" ]; then
    echo "❌ Folder pretrained_pool has not been downloaded. Exiting..."
    exit 1  # Stop the script
fi

make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T6 \
        augment=Proxy \
        seed=1 \
        all

deactivate