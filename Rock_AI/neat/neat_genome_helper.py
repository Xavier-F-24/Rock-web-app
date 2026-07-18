"""Resource-bounded NEAT genome retaining neat-python innovation/speciation logic."""

from __future__ import annotations

import copy

import neat

from .neat_complexity_helper import complexity_within_limits
from .neat_topology_helper import TopologyResourceLimits


class BoundedRecurrentGenome(neat.DefaultGenome):
    resource_limits = TopologyResourceLimits()
    resource_limit_hits = 0

    def mutate(self, config):
        nodes_before = copy.deepcopy(self.nodes)
        connections_before = copy.deepcopy(self.connections)
        super().mutate(config)
        self._rock_output_count = config.num_outputs
        if not complexity_within_limits(self, type(self).resource_limits):
            self.nodes = nodes_before
            self.connections = connections_before
            type(self).resource_limit_hits += 1
