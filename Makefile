COVERAGE=coverage

#install:
#	python3 -m pip install .

#uninstall:
#	python3 -m pip uninstall pando

sa:
	cd tnn_learning; python3 tnn_perform_SA.py

utest:
	$(COVERAGE) run --rcfile=.coveragerc -m pytest .

.PHONY: sa, utest
