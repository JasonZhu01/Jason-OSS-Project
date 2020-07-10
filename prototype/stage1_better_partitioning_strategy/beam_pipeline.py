"""


@author: jzhunybj
"""

import tensorflow as tf
import apache_beam as beam
import copy


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
