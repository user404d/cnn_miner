all: run_classify

install:
	@test -d cnn_env || python3.5 -m venv cnn_env
	@while [ ! -f cnn_env/bin/activate ]; do \
	    echo "waiting for environment..."; \
	    sleep 1; \
	  done;
	@source cnn_env/bin/activate; \
	pip3.5 install -r requirements.txt; \

prep:
	-@mkdir output
	-@tar xzf selected_stories.tar.gz

run_ranking: prep
	@test -f cnn_env/bin/activate || (echo "virtual environment unavailable" && exit 1)
	-@echo "Generating rankings..."
	@source cnn_env/bin/activate; \
	python3.5 generate_rankings.py; \

run_classify: prep
	@test -f cnn_env/bin/activate || (echo "virtual environment unavailable" && exit 1)
	-@echo "Generating classifications..."
	@source cnn_env/bin/activate; \
	python3.5 generate_classifications.py \

clean:
	-@rm -r cnn_env
	-@rm output/*
