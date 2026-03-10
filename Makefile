test:
	pytest

doctest:
	pytest --doctest-modules src/graphcalc -q

test-all:
	pytest
	pytest --doctest-modules src/graphcalc -q
