# Makefile to automate build of autoRS.exe file.
# Aims:
#	- Setup virtual environment with all required packages (requirements.txt)
#	- Run all tests
#	- Use pyinstaller to build .exe file
#	- Delete any artifacts/temporary files

# Variable to define the prerequisite files for build
files = run.py autoRS/__init__.py autoRS/resp_spect.py autoRS/rw.py

# Command that makes file run all subsequent commands
all: dist test

# Test command
test:
	@echo "Test Makefile" $(CURDIR)

# Remove existing dist/build folders and rebuild if out of date
dist: $(files)
	-rd /q /s build dist 2>nul
	-del *.spec 2>nul
	pyinstaller -n autoRS run.py -F

.PHONY: all test
