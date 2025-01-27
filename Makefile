# Variables
VENV_DIR = venv
FLASK_APP = app.py
FLASK_PORT = 5000
NODE_PORT = 3000

# Default target
all: install run

# Create and activate virtual environment, install Python dependencies
$(VENV_DIR)/bin/activate: requirements.txt
	@echo "Setting up Python virtual environment..."
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt
	@touch $(VENV_DIR)/bin/activate

# Install Node.js dependencies
node_modules: package.json
	@echo "Installing Node.js dependencies..."
	npm install
	@touch node_modules

# Run Flask backend
run-flask: $(VENV_DIR)/bin/activate
	@echo "Starting Flask backend..."
	. $(VENV_DIR)/bin/activate && FLASK_APP=$(FLASK_APP) flask run --port=$(FLASK_PORT)

# Run Node.js frontend
run-node: node_modules
	@echo "Starting Node.js frontend..."
	npm start

# Install all dependencies
install: $(VENV_DIR)/bin/activate node_modules

# Run both frontend and backend
run: run-flask run-node

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR) node_modules
