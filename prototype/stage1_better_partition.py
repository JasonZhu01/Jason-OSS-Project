"""
Project: Distributed Batch Inference on large TF models

Goal: Run models with remote ops in Beam

Requirements: TensorFlow 2.x.x + Apache Beam 2.x.x

***IMPORTANT***: We need to import the graph definition to declare PyFunc! 

    Stage 1: working with GraphDef
        a) Create models with remote ops
        b) Partition the graph and save the subgraphs
        c) Arrange the partitioned graphs into a Beam custom PTransform
        d) Save the inference results in PredictionLog format

    Current Implementation:

@author: jzhunybj
"""

import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
import apache_beam as beam
import copy

"""Problem: since we're using PyFunc to mimic remote ops, we need to include
            the file that declares the graphs. Otherwise, the PyFunc ops
            cannot be found/loaded."""
import stage1_harder_graph

    
class GraphPartition(object):
    """A class that supports graph partitioning on TF models with remote ops
    
    The current implementation follows the idea of singular op decomposition.
        In other word, we treat each TF node/op as a subgraph.
        Pro: Very modular
        Con: Prevents TF from doing graph-level optimization (ex: merging nodes)
        
    Settings:
        We have different graphs, including one main graph and several remote graphs.
    
    Functionalities:
         1. Get the execution relations for each graph/subgraph
         2. Partition the graph/subgraphs
         
    Attributes:
        ***Definition: op = graph names = {'main', 'remote_op_a', 'remote_op_b'} in our example***
        op_to_graph_def: {graph name: graph_def}
        op_to_graph: {graph name: graph}
        op_to_execution_info: {graph name: {'graph_placeholder_inputs': a list of node names},
                                            'execution relations': {node name: a list of input node names}}
        op_to_partitioned_graph: {graph_name: {node name: graph_def}}
    """
    
    def __init__(self, op_to_filepath, op_to_outputs):
        """Initialize the op_to_graph_def and op_to_graph mappings"""
        self.op_to_graph_def = {}
        self.op_to_graph = {}
        self.op_to_node_name_to_node_def = {}
        self.op_to_outputs = op_to_outputs
        
        for op in op_to_filepath:
            self.op_to_graph_def[op] = self._get_graph_def(op_to_filepath[op])
            self.op_to_graph[op] = self._get_graph(self.op_to_graph_def[op])
            self.op_to_node_name_to_node_def[op] = self._get_node_def(self.op_to_graph_def[op])
            
    def _get_graph_def(self, filepath):
        """Arg: filepath. Return: graph_def"""
        graph_def = graph_pb2.GraphDef()
        with tf.compat.v1.gfile.FastGFile(filepath, 'rb') as f:
            graph_def.ParseFromString(f.read())
        return graph_def
    
    def _get_graph(self, graph_def):
        """Arg: graph_def. Return: graph"""
        temp = tf.Graph()
        with temp.as_default():
            tf.import_graph_def(graph_def)
            return tf.compat.v1.get_default_graph()
        
    def _get_node_def(self, graph_def):
        node_name_to_node_def = {}
        for node in graph_def.node:
            node_name_to_node_def[node.name] = node
        return node_name_to_node_def
    
        
    def partition(self):
        """Perform graph partitioning"""
        self._get_execution_info_for_all()
        self._create_graphdefs_for_all()
        
    def TEST(self):
        """Show the attributes after the partitioning"""
        for op, execution_relations in self.op_to_execution_info.items():
            print('%s\n%s' % (op, execution_relations))
    
    
    def _check_input_placeholder_op(self, node):
        return node.op == "Placeholder"
    
    def _check_remote_op(self, node):
        return bool(sum([node.name.startswith(prefix) for prefix in self.op_to_graph_def]))
    
    def _bfs_get_remote_op_relations_for_one_graph(self, node_name, execution_relations, node_name_to_node_def):
        """For a remote op, get the remote op names required for the execution.
        
        Arguments:
            node_name: a remote op name
            execution_relations: {node name: a list of input node names}.
        
        Returns:
            A list of children remote ops.
        """
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
                    
                    if self._check_remote_op(input_node):
                        children_remote_op.append(input_node_name)
                    else:
                        queue.append(input_node_name)
        
        return children_remote_op
    
    def _get_execution_info_for_one_graph(self, graph_def, node_name_to_node_def):
        """Get the execution info of a graph"""
        graph_placeholder_inputs = []
        execution_relations = {}
        remote_op_relations = {}
        
        for node in graph_def.node:
            if self._check_input_placeholder_op(node):
                graph_placeholder_inputs.append(node.name)
            
            execution_relations[node.name] = list(node.input)       # converts proto-repeated to list
            
        for node in graph_def.node:
            if self._check_remote_op(node):
                remote_op_relations[node.name] = self._bfs_get_remote_op_relations_for_one_graph(node.name, execution_relations, node_name_to_node_def)
                
        return {'graph_placeholder_inputs': graph_placeholder_inputs,
                'execution_relations': execution_relations,
                'remote_op_relations': remote_op_relations}
    
    def _get_execution_info_for_all(self):
        """Get the execution info for all graphs"""
        self.op_to_execution_info = {}
        
        for op, graph_def in self.op_to_graph_def.items():
            self.op_to_execution_info[op] = self._get_execution_info_for_one_graph(graph_def, self.op_to_node_name_to_node_def[op])
        
    
    
    def _create_placeholder_node(self, dtype, shape, name):
        temp = tf.Graph()
        with temp.as_default():
            placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
            return temp.as_graph_def().node[0]      # The first and the only node  
    
    def _add_placeholder_node(self, subgraph, node, graph):
        """Replace remote op input with a placeholder node"""
        operation = graph.get_operation_by_name('import/%s' % (node.name))
        dtype = operation.outputs[0].dtype
        subgraph.node.append(self._create_placeholder_node(dtype=dtype,
                                                           shape=None,
                                                           name=node.name))
    
    def _add_regular_node(self, subgraph, node):
        subgraph.node.append(node)
        
    def _get_inputs_and_set_of_node_names(self, subgraph, placeholder_inputs, set_of_regular_node_names):
        for node in subgraph.node:
            if self._check_input_placeholder_op(node):
                placeholder_inputs.add(node.name)
            else:
                set_of_regular_node_names.add(node.name)
                
    def _update_graph_visited(self, graph_visited, set_of_regular_node_names):
        for regular_node_name in set_of_regular_node_names:
            graph_visited.add(regular_node_name)

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
            if self._check_remote_op(current_node) or self._check_input_placeholder_op(current_node):
                # Remote op or placeholder input will always be prepared.
                if current_node_name not in subgraph_visited:
                    self._add_placeholder_node(subgraph, current_node, graph)
                    subgraph_visited.add(current_node_name)   
            else:
                # Regular op may be a skip connection and not prepared.
                if current_node_name in graph_visited:
                    skip_connections.add(current_node_name)
                
                elif current_node_name not in subgraph_visited:
                    self._add_regular_node(subgraph, current_node)
                    subgraph_visited.add(current_node_name)
                    queue.extend(execution_relations[current_node_name])
        
        self._get_inputs_and_set_of_node_names(subgraph, inputs, set_of_regular_node_names)
        self._update_graph_visited(graph_visited, set_of_regular_node_names)
                
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
        
        order = Relations(remote_op_relations)
        execution_bundles = []
        graph_visited = set([])
        
        while not order.check_if_finished():
            # Handle one subgraph layer
            remote_ops_one_layer = order.get_next_layer()
            output_node_names = set([])
            for remote_op in remote_ops_one_layer:
                for input_node_name in execution_relations[remote_op]:
                    input_node = node_name_to_node_def[input_node_name]
                    if not self._check_input_placeholder_op(input_node):
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
                                self._add_placeholder_node(subgraph, node, graph)
    
        
        # TEST: subgraph validity
        for execution_bundle in execution_bundles:
            print(execution_bundle)
            
            if not execution_bundle['is_remote_op']:
                graph = tf.Graph()
                with graph.as_default():
                    tf.import_graph_def(execution_bundle['subgraph'])
        
        
        # TEST: node name coverage
        node_names = set([])
        for node in graph_def.node:
            if not self._check_remote_op(node) and not self._check_input_placeholder_op(node):
                node_names.add(node.name)
                
        print("We've visited all the regular nodes inside a graph:", graph_visited == node_names)
    
    
        return execution_bundles
        
    def _create_graphdefs_for_all(self):
        self.op_to_execution_bundles = {}
        
        for op in self.op_to_graph_def:
            self.op_to_execution_bundles[op] = self._create_graphdefs_for_a_graph(op)
                 
    
    def _save_partitioned_results(self, path):
        """---DEVELOPING---
           Save results into files"""
        for op, partitioned_graph in self.op_to_partitioned_graph.items():
            
            for node_name, partitioned_subgraph in partitioned_graph.items():
                node_name = node_name.replace('/', '-')
                print(node_name)
                logdir = '%s/partition/%s/' % (path, op)
                tf.io.write_graph(partitioned_subgraph, logdir, '%s.pb' % node_name, as_text=False)
        
    
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
            

class ExecuteOneSubgraph(beam.DoFn):    
    
    def process(self, element, bundle, graph_name):
        """Executes the smallest unit: one subgraph.
        
        Takes in the graph_def of a subgraph, loads the subgraph into a graph,
            executes the subgraph, and stores the result into element.
        
        Definition of element: a unit of PCollection. In our use case, element
            stores the intermediate results in TF graph execution.
        
        Args:
            element: a unit of PCollection, {graph_name: {computed node: value}}
            bundle: an execution bundle that contains things like graph_def, inputs, and outputs
            graph_name: which graph am I belonging to, ex: 'remote_op_a'
            
        Returns:
            Latest element with the recently computed outputs added.
        """
        self.graph_def = bundle['subgraph']
        self.output_names = []
        for output_name in bundle['outputs']:
            self.output_names.append(self._import_name(output_name))
            
        self.feed_dict = {}
        for input_name in bundle['inputs']:
            input_name = self._import_name(input_name)
            self.feed_dict[input_name] = element[graph_name][input_name]
        
        self._setup()
        output = self._run_inference()
        
        element = copy.deepcopy(element)
        
        for output_name_index in range(len(self.output_names)):
            output_name = self.output_names[output_name_index]
            element[graph_name][output_name] = output[output_name_index]
        
        yield element
    
    def _import_name(self, string):
        return 'import/%s:0' % string
    
    def _setup(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)
        
    def _run_inference(self):
        with tf.compat.v1.Session(graph=self.graph) as sess:
            return sess.run(self.output_names, feed_dict=self.feed_dict)


@beam.ptransform_fn
def ExecuteOneGraph(pcoll,
                    op_to_execution_bundles,
                    op_to_remote_op_name_mapping,
                    graph_name):
    """Compute one graph, for example remote_op_b's graph.
    
    Here, we arrange and execute the partitioned subgraphs in order. In most cases,
        we execute one partitioned subgraph at a time by calling the DoFn BeamSubgraph().
        However, when we encounter a remote op, we recursively call the PTransform
        BeamGraph().
    
    Main assumption:
        The parent graph has set up the placeholder values for the child graph.
        This means that we need to setup the placeholder values (in PColl) for the main graph.
    
    Args:
        pcoll: input PCollection, each unit contains {graph_name: {'computed tensor': value}}
        op_to_execution_bundles: {graph_name: [bundle1, bundle2, ...]}
            bundle: {'subgraph': graph_def, 'inputs': a set of node names,
                     'outputs': a set of node names, 'is_remote_op': a boolean}
        
        # Need improvements
        op_to_remote_op_name_mapping: {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}
        graph_name: which graph am I currently executing
        
    Returns:
        pcoll with the intermediate/end results.
    """
    def _get_op_type(name):
        for graph_name in op_to_execution_bundles:
            if name.startswith(graph_name):
                return graph_name
    
    op_type = _get_op_type(graph_name)
    execution_bundles = op_to_execution_bundles[op_type]
    
    count = 0
    
    for bundle in execution_bundles:
        print(bundle['set_of_node_names'])
        
        if not bundle['is_remote_op']:
            """Executing a subgraph"""
            count += 1
            pcoll = pcoll | str(count) >> beam.ParDo(ExecuteOneSubgraph(),
                                                     bundle,
                                                     graph_name)
            
        else:
            """Executing a remote op, which is another graph"""
            def copy_tensor(element, old_graph, old_tensor, new_graph, new_tensor):
                """Add tensor in pcoll: from a graph to another.
                
                The purpose is to preprocess/postprocess the pcoll before/after
                    the recursive call.
                    
                Preprocess:
                    Prepare for the placeholders nodes in the remote graph.
                Postprocess:
                    Transfer the results of the remote graph back to the parent graph.
                """
                element = copy.deepcopy(element)
                if new_graph not in element:
                    element[new_graph] = {}
                
                element[new_graph][new_tensor] = element[old_graph][old_tensor]
                return element
            
            def remove_finished_graph_info(element, finished_graph):
                element = copy.deepcopy(element)
                del element[finished_graph]
                return element
            
            def _import_name(name):
                return 'import/%s:0' % name
            
            current_graph_name = graph_name
            remote_graph_name = list(bundle['outputs'])[0]      # only one remote op output
            
            """Preprocess"""
            placeholder_name_to_input_name = op_to_remote_op_name_mapping[current_graph_name][remote_graph_name]
            for placeholder_name, input_name in placeholder_name_to_input_name.items():
                count += 1
                pcoll = pcoll | str(count) >> beam.Map(copy_tensor,
                                                       current_graph_name,
                                                       _import_name(input_name),
                                                       remote_graph_name,
                                                       _import_name(placeholder_name))
                
            """Recurse: execute the remote graph"""
            count += 1
            pcoll = pcoll | str(count) >> ExecuteOneGraph(op_to_execution_bundles,
                                                          op_to_remote_op_name_mapping,
                                                          remote_graph_name)
            
            """Postprocess
               Since we're executing a remote op, it will only have one output (current assumption)"""
            remote_op_bundle = op_to_execution_bundles[_get_op_type(remote_graph_name)]
            remote_op_output_name = list(remote_op_bundle[-1]['outputs'])[0]
            count += 1
            pcoll = pcoll | str(count) >> beam.Map(copy_tensor,
                                                   remote_graph_name,
                                                   _import_name(remote_op_output_name),
                                                   current_graph_name,
                                                   _import_name(remote_graph_name))
            
            count += 1
            pcoll = pcoll | str(count) >> beam.Map(remove_finished_graph_info,
                                                   remote_graph_name)
        
    return pcoll



def TEST_Partitioning():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb',
                      }
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    
    partition = GraphPartition(op_to_filename, op_to_outputs)
    partition.partition()
#    partition.TEST()
    
    
def TEST_Subgraph():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb',
                      }
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    
    partition = GraphPartition(op_to_filename, op_to_outputs)
    partition.partition()
    
    feed_dicts_graph_b = [{'remote_op_b': {'import/ids_b1:0': 3, 'import/ids_b2:0': 3}},
                          {'remote_op_b': {'import/ids_b1:0': 10, 'import/ids_b2:0': 10}}]
    
    graph_name = 'remote_op_b'
    output_names = ['import/FloorMod:0']
    bundle = partition.op_to_execution_bundles[graph_name][0]
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
        
        inputs = p | 'read' >> beam.Create(feed_dicts_graph_b)
        
        outputs = inputs | 'Graph' >> beam.ParDo(ExecuteOneSubgraph(),
                                                 bundle,
                                                 graph_name)
        
        class GetOutputs(beam.DoFn):
            def process(self, element, graph_name, output_names):
                result = [element[graph_name][output_name] for output_name in output_names]
                yield result
        
        outputs = outputs | beam.ParDo(GetOutputs(), graph_name, output_names)
        outputs | 'output' >> beam.io.WriteToText('./beam_experiment')
                
        result = p.run()
        result.wait_until_finish()
    
    result_original_model = []
    for feed_dict in feed_dicts_graph_b:
        graph = partition.op_to_graph[graph_name]
        with tf.compat.v1.Session(graph=graph) as sess:
                result_original_model.append(sess.run(output_names, feed_dict[graph_name]))
                
    import subprocess
    result_beam_pipeline = subprocess.check_output(['cat', './beam_experiment-00000-of-00001'])
    
    print('Results from the original model:', result_original_model)
    print('\nResults from the beam pipeline:', result_beam_pipeline)
    
    

def TEST_Execute_Original_Model(graph_name, output_names, feed_dicts):
    """Execute the original model.
    
    Args:
        graph_name: the graph to execute
        output_name: the name of the output inside the graph
        feed_dicts: a list of {graph_name: {placeholder_input: value}}
        
    Returns:
        A list of results
    """
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'} 
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    test = GraphPartition(op_to_filename, op_to_outputs)
    
    results = []
    for feed_dict in feed_dicts:
        graph = test.op_to_graph[graph_name]
        with tf.compat.v1.Session(graph=graph) as sess:
                results.append(sess.run(output_names, feed_dict[graph_name]))
    
    return results


def TEST_BeamPipeline(graph_name, output_names, feed_dicts):
    """Execute the Beam Pipeline.
    
    Args:
        graph_name: the graph to execute
        output_name: the name of the output inside the graph
        feed_dicts: a list of {graph_name: {placeholder_input: value}}
        
    Returns:
        None. Save the result into a file.
    """
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb',
                      }
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    
    test = GraphPartition(op_to_filename, op_to_outputs)
    test.partition()
    
    """Define your feed_dict again.
    
    These relations are stored inside PyFunc, but we don't have the access.
    {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}
    """
    op_to_remote_op_name_mapping = {'main': {'remote_op_a': {'ids_a': 'ids1'},
                                             'remote_op_b': {'ids_b1': 'ids1', 'ids_b2': 'ids2'},
                                             'remote_op_a_1': {'ids_a': 'FloorMod'}},
                                    'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                                                    'remote_op_a_1': {'ids_a': 'ids_b2'}},
                                    }
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
        
        inputs = p | 'read' >> beam.Create(feed_dicts)
    
        outputs = inputs | 'Graph' >> ExecuteOneGraph(test.op_to_execution_bundles,
                                                      op_to_remote_op_name_mapping,
                                                      graph_name)
        
        class GetOutput(beam.DoFn):
            def process(self, element, graph_name, output_names):
                outputs = []
                for output_name in output_names:
                    outputs.append(element[graph_name][output_name])
                yield outputs
            
        outputs = outputs | beam.ParDo(GetOutput(), graph_name, output_names)
        outputs | 'output' >> beam.io.WriteToText('./beam_experiment')
                
        result = p.run()
        result.wait_until_finish()
        

def TEST_Stage1(graph_name):
    feed_dicts_main_graph = [{'main': {'import/ids1:0': 3, 'import/ids2:0': 3}},
                             {'main': {'import/ids1:0': 10, 'import/ids2:0': 10}}]
    
    feed_dicts_graph_b = [{'remote_op_b': {'import/ids_b1:0': 3, 'import/ids_b2:0': 3}},
                          {'remote_op_b': {'import/ids_b1:0': 10, 'import/ids_b2:0': 10}}]
    
    feed_dicts_graph_a = [{'remote_op_a': {'import/ids_a:0': 3}},
                          {'remote_op_a': {'import/ids_a:0': 10}}]
    
    """The testcases are comprised of tests for each graph"""
    testcases = {'main': {'output_names': ['import/Mean:0', 'import/Add:0'],
                          'feed_dicts': feed_dicts_main_graph},
                'remote_op_b': {'output_names': ['import/Add_1:0'],
                                'feed_dicts': feed_dicts_graph_b},               
                'remote_op_a': {'output_names': ['import/embedding_lookup/Identity:0'],
                                'feed_dicts': feed_dicts_graph_a},
                                }
    
    print('\nExecuting original model...')
    result_original_model = TEST_Execute_Original_Model(graph_name, 
                                                        testcases[graph_name]['output_names'],
                                                        testcases[graph_name]['feed_dicts'])
    
    print('\nExecuting the Beam pipeline...')
    TEST_BeamPipeline(graph_name, 
                      testcases[graph_name]['output_names'],
                      testcases[graph_name]['feed_dicts'])
    
    import subprocess
    result_beam_pipeline = subprocess.check_output(['cat', './beam_experiment-00000-of-00001'])
    
    print('Results from the original model:', result_original_model)
    print('\nResults from the beam pipeline:', result_beam_pipeline)
    
        
if __name__ == "__main__":
    """Testcases:
    
    TEST_Partitioning(): test the partitioning functionality
    TEST_Relations(): test the relations functionality
    TEST_Stage1(graph_name): compare the original model with the beam pipeline
    """
    graph_names = ['main', 'remote_op_a', 'remote_op_b']
    TEST_Stage1('main')
    
    
    
    








