ifndef submission
$(error submission is undefined)
endif

build:
	rm -Rf package2
	mkdir package2
	mkdir package2/predictions
	mkdir package2/datasets
	rsync -ar --exclude='**/test_y.npy' datasets/* package2/datasets/
	cp -R evaluation/main.py package2/main.py
	cp -R $(submission)/* package2

run:
	cd package2; python3 main.py

score:
	rm -Rf scoring2
	mkdir scoring2
	mkdir scoring2/labels
	mkdir scoring2/predictions
	rsync -avr --exclude='**/*x.npy' --exclude='**/train*.npy' --exclude='**/valid*.npy'   --include='**/test_y.npy' datasets/* scoring2/labels/
	cp -R package2/predictions scoring2
	cp evaluation/score.py scoring2/score.py
	cd scoring2; python3 score.py

clean:
	rm -Rf scoring2
	rm -Rf package2

zip:
	cd $(submission);  zip -r ../submission.zip *

all: clean build run score