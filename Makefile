all: prep run

prep:
	-@pip3.5 install -r requirements.txt
	-@mkdir output
	-@tar xzvf selected_stories.tar.gz

run: output/
	-@echo "Generating rankings..."
	-@python3.5 generate_rankings.py

clean:
	-@rm output/*
