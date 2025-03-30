# Managing reliability and computational resources using Genetic Algorithms


This project simulates cloud service auto scaling with different traffic patterns and uses genetic algorithms to create optimal scaling decisions. The simulation evaluates strategies based on response time, VM utilization, and SLO compliance.


## Setup Instructions

1. **Create and activate a virtual environment:**
   ```bash
   # Create a virtual environment named .venv
   python -m venv .venv

   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   # Make sure your virtual environment is activated (.venv should appear in your terminal)
   pip install -r requirements.txt
   ```

3. **Set up Jupyter with the virtual environment:**
   ```bash
   # Install ipykernel to register the virtual environment with Jupyter
   pip install ipykernel
   
   # Register the virtual environment as a Jupyter kernel
   python -m ipykernel install --user --name=.venv --display-name="GA Cloud Simulation (venv)"
   ```

4. **Run Jupyter:**
   ```bash
   jupyter notebook
   ```
   - Open `simulation.ipynb` in Jupyter
   - Select Kernel → Change kernel → "Cloud Simulation (venv)"
   - Run cells sequentially to see simulation results

5. **Run tests:**
   ```bash
   python -m unittest test_cloud_simulation.py
   ```

