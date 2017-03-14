all: run_classify

install:
	@test -d cnn_env || python -m venv cnn_env
	@while [ ! -f cnn_env/bin/activate ]; do \
	    echo "waiting for environment..."; \
	    sleep 1; \
	  done;
	@source cnn_env/bin/activate; \
	pip install -r requirements.txt; \

prep:
	-@mkdir output
	-@tar xzf selected_stories.tar.gz

run_ranking: prep
	@test -f cnn_env/bin/activate || (echo "virtual environment unavailable" && exit 1)
	-@echo "Generating rankings..."
	@source cnn_env/bin/activate; \
	python generate_rankings.py; \

run_classify: prep
	@test -f cnn_env/bin/activate || (echo "virtual environment unavailable" && exit 1)
	-@echo "Generating classifications..."
	@source cnn_env/bin/activate; \
	python generate_classifications.py \

run_cluster: prep
	@test -f cnn_env/bin/activate || (echo "virtual environment unavailable" && exit 1)
	-@echo "Generating cluster classifications..."
	@source cnn_env/bin/activate; \
	python generate_clusters.py \

clean:
	-@rm -r cnn_env
	-@rm output/*
