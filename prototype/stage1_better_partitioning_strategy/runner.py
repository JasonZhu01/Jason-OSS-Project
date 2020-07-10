"""


@author: jzhunybj
"""

import partition
import apache_beam as beam
import beam_pipeline


def import_name(name):
    return 'import/%s:0' % name


def run_partition_and_beam(graph_name, op_to_filename, op_to_outputs, op_to_feed_dicts, op_to_remote_op_name_mapping):
    test = partition.GraphPartition(op_to_filename, op_to_outputs)
    test.partition()
    
    options = beam.options.pipeline_options.PipelineOptions()
    with beam.Pipeline(options=options) as p:
        
        inputs = p | 'Load' >> beam.Create(op_to_feed_dicts[graph_name])
    
        outputs = inputs | 'Execute' >> beam_pipeline.ExecuteOneGraph(test.op_to_execution_bundles,
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
    