"""
Test cases.

@author: jzhunybj
"""

import graph_partition
import beam_pipeline
import runner
import subprocess


def TEST_Partitioning():
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb',
                      }
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    
    op_to_graph_def = graph_partition.get_op_to_graph_def(op_to_filename)
    op_to_execution_bundles = graph_partition.partition_all_graphs(op_to_graph_def, op_to_outputs)
    
    for op, execution_bundles in op_to_execution_bundles.items():
        print(op)
        
        for execution_bundle in execution_bundles:
            print(execution_bundle)
            

def TEST(graph_name):
    op_to_filename = {'main': './graphdefs/main_graph.pb',
                      'remote_op_a': './graphdefs/graph_a.pb',
                      'remote_op_b': './graphdefs/graph_b.pb',
                      }
    op_to_outputs = {'main': ['Add'],
                    'remote_op_b': ['Add_1'],
                    'remote_op_a': ['embedding_lookup/Identity'],
                    }
    
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
                                    
    feed_dicts_main_graph = [{'main': {'import/ids1:0': 3, 'import/ids2:0': 3}},
                             {'main': {'import/ids1:0': 10, 'import/ids2:0': 10}}]
    
    feed_dicts_graph_b = [{'remote_op_b': {'import/ids_b1:0': 3, 'import/ids_b2:0': 3}},
                          {'remote_op_b': {'import/ids_b1:0': 10, 'import/ids_b2:0': 10}}]
    
    feed_dicts_graph_a = [{'remote_op_a': {'import/ids_a:0': 3}},
                          {'remote_op_a': {'import/ids_a:0': 10}}]

    op_to_feed_dicts = {'main': feed_dicts_main_graph,
                        'remote_op_b': feed_dicts_graph_b,
                        'remote_op_a': feed_dicts_graph_a}
    
    
    result_original_model = runner.run_original_model(graph_name, 
                                                      op_to_filename, 
                                                      op_to_outputs, 
                                                      op_to_feed_dicts)
    
    runner.run_partition_and_beam(graph_name, 
                                  op_to_filename, 
                                  op_to_outputs, 
                                  op_to_feed_dicts, 
                                  op_to_remote_op_name_mapping)
    result_beam_pipeline = subprocess.check_output(['cat', './beam_experiment-00000-of-00001'])
    
    print('Results from the original model:', result_original_model)
    print('\nResults from the beam pipeline:', result_beam_pipeline)
    
    
if __name__ == '__main__':
    TEST('main')
    
    
    
    
    
    
    