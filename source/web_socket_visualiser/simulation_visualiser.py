import json

from source.simulation import Simulation
from source.web_socket_visualiser import VisualiserServer


class SimulationVisualiser(object):
    def __init__(self, simulation=None, visualiser=None):
        """
        Visualises a simulation instance by piping serialised simulation elements to a visualisation server

        :type simulation: Simulation
        :type visualiser: VisualiserServer
        """
        self.simulation = simulation
        self.visualiser = visualiser

    def update(self):
        inputs = self.visualiser.fetch_messages()
        self.simulation.update(inputs, self.simulation.UPDATE_INTERVAL_MS/1000)
        outputs = []
        for key, element in self.simulation.elements.items():
            d = element.serialise()
            d['id'] = key
            outputs.append(json.dumps(d))

        self.visualiser.send_messages(outputs)


if __name__ == '__main__':
    sim_vis = SimulationVisualiser()
    sim_vis.simulation = Simulation()
    sim_vis.visualiser = VisualiserServer(
        update_callback=sim_vis.update,
        update_interval_ms=Simulation.UPDATE_INTERVAL_MS
    )
    sim_vis.visualiser.start()