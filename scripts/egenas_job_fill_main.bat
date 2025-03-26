@echo off
call .venv\Scripts\activate
make -f Makefile save_folder=C:\Users\woody\iwb3\iwb3021h\THESIS_RESULTS ^
        submission=egenas ^
        mode=T0 ^
        augment=Basic ^
        seed=4 ^
        all
deactivate
