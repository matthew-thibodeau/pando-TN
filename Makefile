COVERAGE=coverage

#install:
#	python3 -m pip install .

#uninstall:
#	python3 -m pip uninstall pando

sa:
	cd ttn_learning; python3 ttn_perform_SA.py $(ARGS)

utest:
	$(COVERAGE) run --rcfile=.coveragerc -m pytest .

rtest:
	cd ttn_learning; python3 ttn_perform_SA.py -L 4 -m 3 -d 1 -s 1234

.PHONY: sa, utest
