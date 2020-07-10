"""


@author: jzhunybj
"""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def get_graph_def(filepath):
    graph_def = graph_pb2.GraphDef()
    with tf.compat.v1.gfile.FastGFile(filepath, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def get_graph(graph_def):
    temp = tf.Graph()
    with temp.as_default():
        tf.import_graph_def(graph_def)
        return tf.compat.v1.get_default_graph()


def get_node_name_to_node_def(graph_def):
    node_name_to_node_def = {}
    for node in graph_def.node:
        node_name_to_node_def[node.name] = node
    return node_name_to_node_def


def check_placeholder_op(node):
    return node.op == "Placeholder"


def check_remote_op(node):
    return node.op == "PyFunc"


def bfs_get_remote_op_children(node_name, execution_relations, node_name_to_node_def):
    """Given a remote op, find the immediate remote op children"""
    queue = [node_name]
    visited = set([node_name])
    children_remote_op = []
    
    while queue:
        current_node_name = queue[0]
        del queue[0]
        
        for input_node_name in execution_relations[current_node_name]:
            if input_node_name not in visited:
                visited.add(input_node_name)
                input_node = node_name_to_node_def[input_node_name]
                
                if check_remote_op(input_node):
                    children_remote_op.append(input_node_name)
                else:
                    queue.append(input_node_name)
    
    return children_remote_op


def create_placeholder_node(dtype, shape, name):
    temp = tf.Graph()
    with temp.as_default():
        placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
        return temp.as_graph_def().node[0]      # The first and the only node  


def add_placeholder_node(subgraph, node, graph):
    """Replace an input node with a placeholder node"""
    operation = graph.get_operation_by_name('import/%s' % (node.name))
    dtype = operation.outputs[0].dtype
    subgraph.node.append(create_placeholder_node(dtype=dtype,
                                                 shape=None,
                                                 name=node.name))

def add_regular_node(subgraph, node):
    subgraph.node.append(node)


def get_inputs_and_regular_nodes(subgraph, placeholder_inputs, set_of_regular_node_names):
    for node in subgraph.node:
        if check_placeholder_op(node):
            placeholder_inputs.add(node.name)
        else:
            set_of_regular_node_names.add(node.name)


def update_graph_visited(graph_visited, set_of_regular_node_names):
    for regular_node_name in set_of_regular_node_names:
        graph_visited.add(regular_node_name)


def TEST_subgraph_validity(execution_bundles):
    for execution_bundle in execution_bundles:        
        if not execution_bundle['is_remote_op']:
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(execution_bundle['subgraph'])


def TEST_regular_node_name_coverage(graph_def, graph_visited):
    node_names = set([])
    for node in graph_def.node:
        if not check_remote_op(node) and not check_placeholder_op(node):
            node_names.add(node.name)
    
    print("We've visited all the regular nodes inside a graph:", graph_visited == node_names)



class Relations(object):
    """A class that outputs layers of a graph"""
    def __init__(self, relations):
        self.relations = relations
        self.processed = set([])
        self.to_be_processed = set(relations.keys())
    
    def check_if_finished(self):
        return not self.to_be_processed
    
    def get_next_layer(self):
        layer_nodes = []
        
        for node in self.to_be_processed:
            node_inputs = set(self.relations[node])
            if node_inputs.issubset(self.processed):
                layer_nodes.append(node)
                
        for node in layer_nodes:
            self.to_be_processed.remove(node)
            self.processed.add(node)
        
        return layer_nodes
    
    def TEST(self):
        while not self.check_if_finished():
            print(self.get_next_layer())
            
    def TEST_hard(self):
        """Draw the graph!"""
        remote_op_relations = {'a1': [], 'a2': [], 'b1': ['a1'], 'b2': ['a1', 'a2'],
                          'c1': ['b1'], 'c2': ['b1', 'a1', 'b2', 'a2']}
        relations = Relations(remote_op_relations)
        relations.TEST()


