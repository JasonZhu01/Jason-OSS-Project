"""
Support a class for graph partitioning.

@author: jzhunybj
"""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2

import partition_helpers


class GraphPartition(object):
    """A class that supports graph partitioning on TF models with remote ops
    
    The current implementation follows the idea of maximal-subgraph partitioning.
        
    Settings:
        We have different graphs, including one main graph and several remote graphs.
    
    Functionalities:
         1. Get the execution relations for each graph/subgraph
         2. Partition the graph/subgraphs
         
    Attributes:
        ***Definition: op = graph names = {'main', 'remote_op_a', 'remote_op_b'} in our example***
        op_to_graph_def: {graph name: graph_def}
        op_to_graph: {graph name: graph}
        op_to_execution_info: {graph name: {'execution relations': {node name: a list of input node names}}
        op_to_partitioned_graph: {graph_name: {node name: graph_def}}
    """
    
    def __init__(self, op_to_filepath, op_to_outputs):
        """Initialize the op_to_graph_def and op_to_graph mappings"""
        self.op_to_graph_def = {}
        self.op_to_graph = {}
        self.op_to_node_name_to_node_def = {}
        self.op_to_outputs = op_to_outputs
        
        for op in op_to_filepath:
            self.op_to_graph_def[op] = partition_helpers.get_graph_def(op_to_filepath[op])
            self.op_to_graph[op] = partition_helpers.get_graph(self.op_to_graph_def[op])
            self.op_to_node_name_to_node_def[op] = partition_helpers.get_node_name_to_node_def(self.op_to_graph_def[op])
    
        
    def partition(self):
        self._get_execution_info_for_all()
        self._create_graphdefs_for_all()
        
    
    def _get_execution_info_for_one_graph(self, graph_def, node_name_to_node_def):
        execution_relations = {}
        remote_op_relations = {}
        
        for node in graph_def.node:            
            execution_relations[node.name] = list(node.input)       # converts proto-repeated to list
            
            if partition_helpers.check_remote_op(node):
                remote_op_relations[node.name] = partition_helpers.bfs_get_remote_op_children(node.name, execution_relations, node_name_to_node_def)
                
        return {'execution_relations': execution_relations,
                'remote_op_relations': remote_op_relations}
    
    def _get_execution_info_for_all(self):
        self.op_to_execution_info = {}
        
        for op, graph_def in self.op_to_graph_def.items():
            self.op_to_execution_info[op] = self._get_execution_info_for_one_graph(graph_def, self.op_to_node_name_to_node_def[op])
                

    def _partition_for_one_subgraph_layer(self, graph_visited, graph_def, graph, outputs, execution_relations, node_name_to_node_def):
        """Perform a modified BFS traversal for partitioning
        
        Arguments:
            graph_visited: a set of regular node that we have visited
            graph_def: a GraphDef
            outputs: outputs required for this subgraph
            execution_relations: {node_name: a list of input node_names}
            node_name_to_node_def: {node_name: node_def}
            
        Returns:
            {'subgraph': a graph_def, 'inputs': a set of nodes, 'outputs': a set of nodes,
             'set_of_node_names': a set of node names, 'is_remote_op': a boolean status}
        """
        subgraph = graph_pb2.GraphDef()
        subgraph.versions.CopyFrom(graph_def.versions)
        subgraph.library.CopyFrom(graph_def.library)
        
        inputs = set([])
        outputs = set(outputs)
        set_of_regular_node_names = set([])
        is_remote_op = False
        skip_connections = set([])
        
        queue = list(outputs)
        subgraph_visited = set([])
        
        while queue:
            current_node_name = queue[0]
            current_node = node_name_to_node_def[current_node_name]
            del queue[0]
            
            """Three types of ops: regular, remote, and graph placeholder input"""
            if partition_helpers.check_remote_op(current_node) or partition_helpers.check_placeholder_op(current_node):
                # Remote op or placeholder input will always be prepared.
                if current_node_name not in subgraph_visited:
                    partition_helpers.add_placeholder_node(subgraph, current_node, graph)
                    subgraph_visited.add(current_node_name)   
            else:
                # Regular op may be a skip connection and not prepared.
                if current_node_name in graph_visited:
                    skip_connections.add(current_node_name)
                
                elif current_node_name not in subgraph_visited:
                    partition_helpers.add_regular_node(subgraph, current_node)
                    subgraph_visited.add(current_node_name)
                    queue.extend(execution_relations[current_node_name])
        
        partition_helpers.get_inputs_and_regular_nodes(subgraph, inputs, set_of_regular_node_names)
        partition_helpers.update_graph_visited(graph_visited, set_of_regular_node_names)
                
        return {'subgraph': subgraph, 
                'inputs': inputs, 
                'outputs': outputs,
                'set_of_node_names': set_of_regular_node_names, 
                'is_remote_op': is_remote_op,
                'skip_connections': skip_connections}
                        
    
    def _partition_for_one_remote_op_layer(self, remote_op_names, execution_relations):
        """Construct structures for remote op"""
        list_of_bundles = []
        for remote_op_name in remote_op_names:
            bundle = {'subgraph': None, 
                      'inputs': set(execution_relations[remote_op_name]), 
                      'outputs': set([remote_op_name]), 
                      'set_of_node_names': set([remote_op_name]), 
                      'is_remote_op': True, 
                      'skip_connections': None}
            list_of_bundles.append(bundle)
        
        return list_of_bundles
    
    
    def _create_graphdefs_for_a_graph(self, op):
        graph_def = self.op_to_graph_def[op]
        graph = self.op_to_graph[op]
        remote_op_relations = self.op_to_execution_info[op]['remote_op_relations']
        execution_relations = self.op_to_execution_info[op]['execution_relations']
        node_name_to_node_def = self.op_to_node_name_to_node_def[op]
        graph_outputs = self.op_to_outputs[op]
        
        order = partition_helpers.Relations(remote_op_relations)
        execution_bundles = []
        graph_visited = set([])
        
        while not order.check_if_finished():
            # Handle one subgraph layer
            remote_ops_one_layer = order.get_next_layer()
            output_node_names = set([])
            for remote_op in remote_ops_one_layer:
                for input_node_name in execution_relations[remote_op]:
                    input_node = node_name_to_node_def[input_node_name]
                    if not partition_helpers.check_placeholder_op(input_node):
                        output_node_names.add(input_node_name)
            
            if output_node_names:
                subgraph_bundle = self._partition_for_one_subgraph_layer(graph_visited, 
                                                                         graph_def,
                                                                         graph,
                                                                         output_node_names, 
                                                                         execution_relations,
                                                                         node_name_to_node_def)
                execution_bundles.append(subgraph_bundle)
            
            # Handle one remote op layer
            remote_op_bundles = self._partition_for_one_remote_op_layer(remote_ops_one_layer, execution_relations)
            execution_bundles.extend(remote_op_bundles)
            
        # Handle the last subgraph layer
        output_node_names = set(graph_outputs)
        subgraph_bundle = self._partition_for_one_subgraph_layer(graph_visited, 
                                                                 graph_def,
                                                                 graph,
                                                                 output_node_names, 
                                                                 execution_relations,
                                                                 node_name_to_node_def)
        execution_bundles.append(subgraph_bundle)
        
        # Handle skip connections
        for current_bundle_index in range(len(execution_bundles)):
            current_bundle = execution_bundles[current_bundle_index]
            
            if not current_bundle['is_remote_op']:
                for skip_node_name in current_bundle['skip_connections']:
                    for previous_bundle_index in range(current_bundle_index):
                        previous_bundle = execution_bundles[previous_bundle_index]
                        
                        if not previous_bundle['is_remote_op']:
                            if skip_node_name in previous_bundle['set_of_node_names']:
                                previous_bundle['outputs'].add(skip_node_name)
                                current_bundle['inputs'].add(skip_node_name)
                                subgraph = current_bundle['subgraph']
                                node = node_name_to_node_def[skip_node_name]
                                partition_helpers.add_placeholder_node(subgraph, node, graph)
    
        
        partition_helpers.TEST_subgraph_validity(execution_bundles)
        partition_helpers.TEST_regular_node_name_coverage(graph_def, graph_visited)   
    
        return execution_bundles
        
    def _create_graphdefs_for_all(self):
        self.op_to_execution_bundles = {}
        
        for op in self.op_to_graph_def:
            self.op_to_execution_bundles[op] = self._create_graphdefs_for_a_graph(op)
                 
    
    def _save_partitioned_results(self, saving_directory):
        """Not Implemented"""
        pass


