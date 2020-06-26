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
        a) 1 example
        b) Singular op decomposition: treat each node as a subgraph
        c) Recursive PTransform
        d) Not implemented yet

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
import stage1_graph

    
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
                                            'execution relations': {node name: a list of node names}}
        op_to_partitioned_graph:
    
    """
    
    def __init__(self, op_to_filepath):
        """Initialize the op_to_graph_def and op_to_graph mappings"""
        self.op_to_graph_def = {}
        self.op_to_graph = {}
        
        for op in op_to_filepath:
            self.op_to_graph_def[op] = self._get_graph_def(op_to_filepath[op])
            self.op_to_graph[op] = self._get_graph(self.op_to_graph_def[op])
            
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
    
    
    def partition(self):
        """Perform graph partitioning"""
        self._get_execution_info_for_all()
        self._create_graphdefs_for_all()
        
    def show_info(self):
        """Show the attributes after the partitioning"""
        for op, execution_relations in self.op_to_execution_info.items():
            print('%s\n%s' % (op, execution_relations))
        
        for op, partitioned_graph in self.op_to_partitioned_graph.items():
            print('%s:' % op)
            for node_name, partitioned_subgraph in partitioned_graph.items():
                print(node_name)
    
    
    def _check_input_placeholder_op(self, node):
        return node.op == "Placeholder"
    
    def _get_execution_info_for_one_graph(self, graph_def):
        """Get the placeholder inputs and the execution info of a graph"""
        graph_placeholder_inputs = []
        execution_relations = {}
        
        for node in graph_def.node:
            if self._check_input_placeholder_op(node):
                graph_placeholder_inputs.append(node.name)
            
            execution_relations[node.name] = list(node.input)       # converts proto-repeated to list
                
        return {'graph_placeholder_inputs': graph_placeholder_inputs,
                'execution_relations': execution_relations}
    
    def _get_execution_info_for_all(self):
        """Get the execution info, including the placeholder inputs and the execution relations"""
        self.op_to_execution_info = {}
        
        for op, graph_def in self.op_to_graph_def.items():
            self.op_to_execution_info[op] = self._get_execution_info_for_one_graph(graph_def)
        
    
    def _check_remote_op(self, node_name):
        return bool(sum([node_name.startswith(prefix) for prefix in self.op_to_graph_def]))
    
    def _create_placeholder_node(self, dtype, shape, name):
        temp = tf.Graph()
        with temp.as_default():
            placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
            return temp.as_graph_def().node[0]      # The first and the only node
            
    """Cautious!
           Reason: removing colocated attr allows the partitioned graph under 
                   the op-level decomposition to work.
           Concern: not sure about the potential impacts."""
    def _remove_colocated_attr(self, node):
        if '_class' in node.attr:
            del node.attr['_class']
        return node
    
    def _remove_prefix_postfix(self, node_name):
        """Remove the prefix "import/" and the postfix ":0" """
        return node_name[7:-2]
        
    def _create_graphdef_for_a_node(self, graph, node, versions, library):
        """Create a partitioned graphdef for a node"""
        graph_def = graph_pb2.GraphDef()
        graph_def.versions.CopyFrom(versions)
        graph_def.library.CopyFrom(library)
        
        node = self._remove_colocated_attr(node) 
        
        current_node = graph.get_operation_by_name('import/%s' % (node.name))
        # Here, we add the input nodes as placeholder nodes into the subgraph
        for input_node in current_node.inputs:
            graph_def.node.append(self._create_placeholder_node(dtype=input_node.dtype,
                                                                shape=None,
                                                                name=self._remove_prefix_postfix(input_node.name)))  
        graph_def.node.append(node)
        return graph_def
                
    def _create_graphdefs_for_a_graph(self, graph_def, graph):
        """Create partitioned graphdefs for a graph"""
        node_to_graph_def = {}
        for node in graph_def.node:
            node_to_graph_def[node.name] = self._create_graphdef_for_a_node(graph,
                                                                            node,
                                                                            graph_def.versions,
                                                                            graph_def.library)
        return node_to_graph_def
    
    def _create_graphdefs_for_all(self):
        """Partition all the graphs"""
        self.op_to_partitioned_graph = {}
        
        for op in self.op_to_graph_def:
            graph_def = self.op_to_graph_def[op]
            graph = self.op_to_graph[op]
            
            self.op_to_partitioned_graph[op] = self._create_graphdefs_for_a_graph(graph_def, graph)
                 
    
    def _save_partitioned_results(self, path):
        """---DEVELOPING---
           Save results into files"""
        for op, partitioned_graph in self.op_to_partitioned_graph.items():
            
            for node_name, partitioned_subgraph in partitioned_graph.items():
                node_name = node_name.replace('/', '-')
                print(node_name)
                logdir = '%s/partition/%s/' % (path, op)
                tf.io.write_graph(partitioned_subgraph, logdir, '%s.pb' % node_name, as_text=False)
        

class BeamSubgraph(beam.DoFn):    
    
    def process(self, element, graph_def, input_names, output_name, graph_name):
        """Executes the smallest unit: one subgraph.
        
        Takes in the graph_def of a subgraph, loads the subgraph into a graph,
            executes the subgraph, and stores the result into element.
        
        Definition of element: a unit of PCollection. In our use case, element
            stores the intermediate results in TF graph execution.
        
        Args:
            element: a unit of PCollection, {graph_name: {'a computed node': value}}
            graph_def: side-input, graph_def of a partitioned graph
            input_names: side-input, inputs of the subgraph (can be empty)
            output_name: side-input, output of the subgrapg
            graph_name: which graph am I belonging to, ex: 'remote_op_a'
            
        Returns:
            Latest element with the recently computed node added.
        """
        self.graph_def = graph_def
        self.output_name = self._convert_name(output_name)
        self.feed_dict = {}
        for input_name in input_names:
            input_name = self._convert_name(input_name)
            self.feed_dict[input_name] = element[graph_name][input_name]
        
        self._setup()
        output = self._run_inference()
        
#        element = copy.deepcopy(element)
        element[graph_name][self.output_name] = output[0]   # The first and the only output tensor
        
        yield element
    
    def _convert_name(self, string):
        return 'import/%s:0' % string
    
    def _setup(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)
        
    def _run_inference(self):
        with tf.compat.v1.Session(graph=self.graph) as sess:
            return sess.run([self.output_name], feed_dict=self.feed_dict)


class Relations:  
    """A class that outputs the order of execution.
    
    Methods:
        check_if_finished(): check if we completed the execution of the entire graph
        find_next(): find the next thing to execute, returns None if finished
        TEST(): for debugging use only
        
    Attributes:
        relations: execution relations of a graph, {node_name: inputs}
        processed: a set that contains nodes that have been processed
        to_be_processed: a set that contains nodes that haven't been processed
    """
    
    def __init__(self, relations, inputs=[]):
        self.relations = relations
        self.processed = set(inputs)
        self.to_be_processed = set(relations.keys())
        for processed in inputs:
            # Assume that inputs have been processed
            if processed in self.to_be_processed:
                self.to_be_processed.remove(processed)
        
    def check_if_finished(self):
        return not self.to_be_processed

    def find_next(self):
        if not self.check_if_finished():
            for elem in self.to_be_processed:
                # If all the inputs were processed
                inputs = set(self.relations[elem])
                if inputs.issubset(self.processed):
                    self.to_be_processed.remove(elem)
                    self.processed.add(elem)
                    return elem
        return None
    
    def TEST(self):
        print(self.relations, self.to_be_processed, self.processed)
        for i in range(len(self.to_be_processed)):
            print(self.find_next())
            

@beam.ptransform_fn
def BeamGraph(pcoll, op_to_partitioned_graph, op_to_execution_info, graph_name, feed_dict, op_to_outputs):
    """
    Assume the parent graph has set up the placeholder values for me
    Inputs:
        pcoll: input PCollection, {graph_name: {tensors ready to use}}
        op_to_partitioned_graph: {graph_name: {node_name: subgraph}}
        op_to_execution_info: {graph_name: {'graph_placeholder_inputs': a list,
                                                 'execution_relations': {node_name: a list of input_names}}}
        graph_name: which graph am I currently executing
        feed_dict: {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}
        op_to_outputs: {graph_name: a list of output names}
    """
    def _get_op_type(remote_graph_name):
        for graph_name in op_to_partitioned_graph:
            if remote_graph_name.startswith(graph_name):
                return graph_name
            
    
    partitioned_graph = op_to_partitioned_graph[_get_op_type(graph_name)]
    execution_info = op_to_execution_info[_get_op_type(graph_name)]
    
    relations = Relations(execution_info['execution_relations'],
                          execution_info['graph_placeholder_inputs'])
    execution_relations = execution_info['execution_relations']
    
    count = 0
    
    while not relations.check_if_finished():
        
        def _check_node_remote_op(node):
            return node.op == 'PyFunc'
        
        current_node_name = relations.find_next()
        current_node = partitioned_graph[current_node_name].node[-1]
        print(current_node_name, _check_node_remote_op(current_node))
        
        if not _check_node_remote_op(current_node):
            """We're executing a node!"""
            graph_def = partitioned_graph[current_node_name]
            input_names = execution_relations[current_node_name]
            output_name = current_node_name
            graph_name = graph_name
            
            count += 1
            pcoll = pcoll | str(count) >> beam.ParDo(BeamSubgraph(),
                                                     graph_def,
                                                     input_names,
                                                     output_name,
                                                     graph_name)
            
        else:
            """We're executing a remote op subgraph!"""                                        
            def add_tensor(element, old_graph, old_tensor, new_graph, new_tensor):
                if new_graph not in element:
                    element[new_graph] = {}
                
                element[new_graph][new_tensor] = element[old_graph][old_tensor]
                print(type(element), len(element))
                
                return element
            
            def remove_executed_graph_info(element, graph):
                del element[graph]
                return element

                    
            def _import_name(name):
                return 'import/%s:0' % name
            
            """Prepare for the inputs"""
            current_graph_name = graph_name
            remote_graph_name = current_node_name
            
            input_names = execution_relations[current_node_name]
            remote_graph_placeholder_names = op_to_execution_info[_get_op_type(remote_graph_name)]['graph_placeholder_inputs']
            placeholder_name_to_input_name = feed_dict[current_graph_name][remote_graph_name]
            
            assert set(input_names) == set(placeholder_name_to_input_name.values())
            assert set(remote_graph_placeholder_names) == set(placeholder_name_to_input_name.keys())
            
            """Get the placeholders ready"""
            for placeholder_name, input_name in placeholder_name_to_input_name.items():
                count += 1
                pcoll | str(count) >> beam.Map(print)
                count += 1
                pcoll = pcoll | str(count) >> beam.Map(add_tensor, 
                                                         current_graph_name, 
                                                         _import_name(input_name), 
                                                         remote_graph_name, 
                                                         _import_name(placeholder_name))
            
            """Recurse"""
            count += 1
            pcoll = pcoll | str(count) >> BeamGraph(op_to_partitioned_graph, op_to_execution_info,
                                                    remote_graph_name, feed_dict, op_to_outputs)
            
            """Get the output"""
            output_names = op_to_outputs[_get_op_type(remote_graph_name)]
            for output_name in output_names:
                count += 1
                pcoll = pcoll | str(count) >> beam.Map(add_tensor,
                                                         remote_graph_name,
                                                         _import_name(output_name),
                                                         current_graph_name,
                                                         _import_name(remote_graph_name))
            
#            count += 1
#            pcoll = pcoll | str(count) >> beam.Map(remove_executed_graph_info,
#                                                   remote_graph_name)
            
                    
    return pcoll     # return is better than yield here, why?



def TEST_Partitioning():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'}
    
    partition = GraphPartition(op_to_filename)
    
    print(partition._check_remote_op('remote_op_a_3'))
    partition._get_execution_info_for_all()
    
    print(partition._create_placeholder_node(tf.int16, None, 'test'))
    partition._create_graphdefs_for_all()

    partition._save_partitioned_results('.')
    
    
    partition = GraphPartition(op_to_filename)
    partition.partition()
    
    test = Relations(partition.op_to_execution_info['remote_op_b']['execution_relations'],
                     partition.op_to_execution_info['remote_op_b']['graph_placeholder_inputs'])
    test.check()

    
def TEST_BeamSubgraph():
    def create_graphdef():
        with tf.compat.v1.Session() as sess:
            a = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='a')
            b = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='b')
            c = tf.add(a, b, name='c')
            print(sess.graph.get_operations()[2].outputs[0].name)
            return sess.graph_def
        
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
        
        graph_def = create_graphdef()
        test = p | 'read' >> beam.Create([
                {'main': {'import/a:0': 1.0, 'import/b:0': 2.0}},
                {'main': {'import/a:0': 2.0, 'import/b:0': 3.0}},
                 ])
        
        class getOutput(beam.DoFn):
            def process(self, element, output):
                return element[output]
            
        output = (
                  test 
                  | 'operate' >> beam.ParDo(BeamSubgraph(), graph_def, ['a', 'b'], 'c', 'main')
#                  | 'get_output' >> beam.ParDo(getOutput(), 'import/c:0')
                 )
            
        output | 'output' >> beam.io.WriteToText('./beam_experiment')
        
        result = p.run()
        

def TEST_BeamGraph_Graph_A():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'}
    
    partition = GraphPartition(op_to_filename)
    partition.partition()
    
    print(list(partition.op_to_graph.keys()))
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
    
        graph_name = 'remote_op_a'
        
        test = p | 'read' >> beam.Create([
                {graph_name: {'import/ids_a:0': 3}},
                {graph_name: {'import/ids_a:0': 10}},
                 ])
    
        test | 'start' >> beam.Map(print)
    
        output = test | 'Graph' >> BeamGraph(partition.op_to_partitioned_graph, partition.op_to_execution_info, 
                                             graph_name, {'remote_op_a': {}}, {'remote_op_a': {}})
        
        class PrintOne(beam.DoFn):
            def process(self, element):
                print(element)
        
        output | 'print' >> beam.ParDo(PrintOne())
        output | 'output' >> beam.io.WriteToText('./beam_experiment')
        
        result = p.run()
        result.wait_until_finish()
        

def TEST_BeamGraph_B():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'}
    
    """Define your feed_dict again
    {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}"""
    feed_dict = {'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                                 'remote_op_a_1': {'ids_a': 'ids_b2'}}
                 }
    
    op_to_outputs = {'remote_op_b': ['Add_1'],
                     'remote_op_a': ['embedding_lookup/Identity']}
    
    partition = GraphPartition(op_to_filename)
    partition.partition()
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
    
        graph_name = 'remote_op_b'
        
        test = p | 'read' >> beam.Create([
                {graph_name: {'import/ids_b1:0': 3, 'import/ids_b2:0': 3}},
                {graph_name: {'import/ids_b1:0': 10, 'import/ids_b2:0': 10}},
                 ])
        
        test | 'start' >> beam.Map(print)
    
        output = test | 'Graph' >> BeamGraph(partition.op_to_partitioned_graph, partition.op_to_execution_info, 
                                             graph_name, feed_dict, op_to_outputs)
        
        class GetOutput(beam.DoFn):
            def process(self, element, graph_name, output_name):
                yield element[graph_name][output_name]
            
        output = output | beam.ParDo(GetOutput(), 'remote_op_b', 'import/Add_1:0')
        output | 'output' >> beam.io.WriteToText('./beam_experiment')
        
        output | 'print' >> beam.Map(print)
        
        
        result = p.run()
        result.wait_until_finish()
        

def TEST_BeamGraph_Main():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'}
    
    """Define your feed_dict again
    {graph_name: {remote op name: {placeholder name inside subgraph: input name}}}"""
    feed_dict = {'main': {'remote_op_a': {'ids_a': 'ids1'},
                          'remote_op_b': {'ids_b1': 'ids1', 'ids_b2': 'ids2'}},
                 'remote_op_b': {'remote_op_a': {'ids_a': 'FloorMod'},
                                 'remote_op_a_1': {'ids_a': 'ids_b2'}}
                 }
    
    op_to_outputs = {'main': ['Mean'],
                     'remote_op_b': ['Add_1'],
                     'remote_op_a': ['embedding_lookup/Identity']}
    
    partition = GraphPartition(op_to_filename)
    partition.partition()
    
    print(list(partition.op_to_graph.keys()))
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
    
        graph_name = 'main'
        
        test = p | 'read' >> beam.Create([
                {graph_name: {'import/ids1:0': 3, 'import/ids2:0': 3}},
                {graph_name: {'import/ids1:0': 10, 'import/ids2:0': 10}},
                 ])
        
        test | 'start' >> beam.Map(print)
    
        output = test | 'Graph' >> BeamGraph(partition.op_to_partitioned_graph, partition.op_to_execution_info, 
                                             graph_name, feed_dict, op_to_outputs)
        
        class GetOutput(beam.DoFn):
            def process(self, element, graph_name, output_name):
                yield element[graph_name][output_name]
            
        output = output | beam.ParDo(GetOutput(), 'main', 'import/Mean:0')
        
        output | 'output' >> beam.io.WriteToText('./beam_experiment')
        
        output | 'print' >> beam.Map(print)
        
        
        result = p.run()
        result.wait_until_finish()


        
if __name__ == "__main__":
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb'}
    
    tf.random.set_seed(5)
    
#    TEST_BeamGraph_Graph_A()
    
    # Calling graph_a
#    tf.random.set_seed(5)
#    test = GraphPartition(directory, op_to_graph)
#    with tf.compat.v1.Session(graph=test.op_to_graph['remote_op_a']) as sess:
#        print(sess.run('import/embedding_lookup/Identity:0', feed_dict={'import/ids_a:0': 3}))
    
    
#    TEST_BeamGraph_B()
    
    # Calling main
    test = GraphPartition(op_to_filename)
    with tf.compat.v1.Session(graph=test.op_to_graph['remote_op_b']) as sess:
        print(sess.run('import/Add_1:0', feed_dict={'import/ids_b1:0': 3, 'import/ids_b2:0': 3}))
#    
    
    TEST_BeamGraph_Main()
    
    # Calling main
#    test = GraphPartition(directory, op_to_graph)
    with tf.compat.v1.Session(graph=test.op_to_graph['main']) as sess:
        print(sess.run('import/remote_op_b:0', feed_dict={'import/ids1:0': 3, 'import/ids2:0': 3}))
    
    
    with tf.compat.v1.Session(graph=test.op_to_graph['main']) as sess:
        print(sess.run('import/Mean:0', feed_dict={'import/ids1:0': 3, 'import/ids2:0': 3}))
        print(sess.run('import/Mean:0', feed_dict={'import/ids1:0': 10, 'import/ids2:0': 10}))
    
    
    


















