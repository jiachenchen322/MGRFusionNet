.PHONY: example simulation-quick simulation-standard

example:
	bash examples/run_simulated_example.sh

simulation-quick:
	bash examples/run_simulated_example.sh

simulation-standard:
	if [ -x .venv/bin/python ]; then \
		.venv/bin/python scripts/simulation_study.py --preset standard --setting both --out results/simulation/simulation_results.txt; \
	else \
		python3 scripts/simulation_study.py --preset standard --setting both --out results/simulation/simulation_results.txt; \
	fi
