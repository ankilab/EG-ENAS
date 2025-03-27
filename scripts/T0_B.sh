#Low cost EG-ENAS
#(RandomErasing + RandomCrop + RandomHflip) augmentation

if [ ! -d "datasets" ]; then
    printf "❌ Datasets folder has not been created. Exiting...\n"
    exit 1  # Stop the script
fi

source .venv/bin/activate
make -f Makefile save_folder=EGENAS_RESULTS \
        submission=egenas \
        mode=T0 \
        augment=Basic \
        seed=1 \
        all

deactivate