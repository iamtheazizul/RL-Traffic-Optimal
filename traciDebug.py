import traci
Sumo_config = ['sumo-gui', '-c', 'simulation_run_rl.sumocfg', '--step-length', '0.5']
traci.start(Sumo_config)
try:
    for step in range(10):
        traci.simulationStep()
        print(f"Step {step}, Time: {traci.simulation.getTime()}")
        print("Traffic Lights:", traci.trafficlight.getIDList())
        print("Detectors:", traci.lanearea.getIDList())
    traci.close()
except traci.exceptions.TraCIException as e:
    print(f"Error: {e}")