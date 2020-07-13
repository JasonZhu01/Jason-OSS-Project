"""
Two runners:
    1. run_partition_and_beam()
    2. run_original_model()

@author: jzhunybj
"""

import tensorflow as tf
import apache_beam as beam
import beam_pipeline
import graph_partition

"""Problem: since we're using PyFunc to mimic remote ops, we need to include
            the file that declares the graphs. Otherwise, the PyFunc ops
            cannot be found/loaded."""
import create_complex_graph


def import_name(name):
    return 'import/%s:0' % name


def run_partition_and_beam(graph_name, op_to_filename, op_to_outputs, op_to_feed_dicts, op_to_remote_op_name_mapping):
    op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
    op_to_execution_bundles = graph_partition.partition_all_graphs(op_to_graph_def, op_to_outputs)
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
        
        inputs = p | 'Load' >> beam.Create(op_to_feed_dicts[graph_name])
    
        outputs = inputs | 'Execute' >> beam_pipeline.ExecuteOneGraph(op_to_execution_bundles,
                                                                      op_to_remote_op_name_mapping,
                                                                      graph_name)
        
        class GetOutput(beam.DoFn):
            def process(self, element, graph_name, output_names):
                outputs = []
                for output_name in output_names:
                    outputs.append(element[graph_name][import_name(output_name)])
                yield outputs
            
        outputs = outputs | beam.ParDo(GetOutput(), graph_name, op_to_outputs[graph_name])
        outputs | 'output' >> beam.io.WriteToText('./beam_experiment')
                
        result = p.run()
        result.wait_until_finish()


def run_original_model(graph_name, op_to_filename, op_to_outputs, op_to_feed_dicts):
    op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
    graph_def = op_to_graph_def[graph_name]
    graph = graph_partition.get_graph(graph_def)
    
    graph_outputs = []
    for output_name in op_to_outputs[graph_name]:
        graph_outputs.append(import_name(output_name))
    feed_dicts = op_to_feed_dicts[graph_name]
    
    results = []
    for feed_dict in feed_dicts:
        with tf.compat.v1.Session(graph=graph) as sess:
                results.append(sess.run(graph_outputs, feed_dict[graph_name]))
    
    return results
    
    
    
    
    
    
    
    
    
    
    
    