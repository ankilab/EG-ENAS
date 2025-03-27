ifndef submission
  $(error "submission is undefined")
endif
ifndef save_folder
  $(error "save_folder is undefined")
endif
ifndef mode
  $(error "mode is undefined")
endif
ifndef augment
  $(error "augment is undefined")
endif
ifndef pretrained_pool_path
  $(error "pretrained pool path is undefined")
endif
ifdef only_processor
    ONLY_PROCESSOR_FLAG=--only_processor
else
    ONLY_PROCESSOR_FLAG=
endif

build:
	rm -Rf $(save_folder)/package
	mkdir -p $(save_folder)
	mkdir $(save_folder)/package
	mkdir $(save_folder)/package/augmentations_test
	mkdir $(save_folder)/package/predictions
	mkdir $(save_folder)/package/datasets
	rsync -ar --exclude='**/test_y.npy' datasets/* $(save_folder)/package/datasets/
	cp -R evaluation/main.py $(save_folder)/package/main.py
	cp -R $(submission)/* $(save_folder)/package

run:
	cd $(save_folder)/package; python3 main.py --mode $(mode) --select_augment $(augment) --seed $(seed) --pretrained_pool_path $(pretrained_pool_path) $(ONLY_PROCESSOR_FLAG)

score:
	rm -Rf $(save_folder)/scoring
	mkdir $(save_folder)/scoring
	mkdir $(save_folder)/scoring/labels
	mkdir $(save_folder)/scoring/predictions
	rsync -avr --exclude='**/*x.npy' --exclude='**/train*.npy' --exclude='**/valid*.npy'   --include='**/test_y.npy' datasets/* $(save_folder)/scoring/labels/
	cp -R $(save_folder)/package/predictions $(save_folder)/scoring
	cp evaluation/score.py $(save_folder)/scoring/score.py
	cd $(save_folder)/scoring; python3 score.py

clean:
	rm -Rf $(save_folder)/scoring
	rm -Rf $(save_folder)/package

zip:
	cd $(submission);  zip -r ../submission.zip *

all: clean build run score