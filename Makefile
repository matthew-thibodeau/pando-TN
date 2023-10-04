COVERAGE=coverage

#install:
#	python3 -m pip install .

#uninstall:
#	python3 -m pip uninstall pando

sa:
	cd ttn_learning; python3 ttn_perform_SA.py $(ARGS)

utest:
	$(COVERAGE) run --rcfile=.coveragerc -m pytest .

.PHONY: sa, utest
