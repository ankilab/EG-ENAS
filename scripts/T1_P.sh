#Random Search
source .venv/bin/activate

if [ ! -d "pretrained_pool" ]; then
    echo "❌ Folder pretrained_pool has not been downloaded. Exiting..."
    exit 1  # Stop the script
fi

make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T1 \
        augment=Proxy \
        seed=1 \
        pretrained_pool_path=pretrained_pool/ \
        all

deactivate