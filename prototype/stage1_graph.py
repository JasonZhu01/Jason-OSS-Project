#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:05:25 2020

@author: jzhunybj
"""


import tensorflow as tf
import numpy as np


tf.compat.v1.disable_eager_execution()  # Disable eager mode

N = 1000    # number of embeddings
NDIMS = 16  # dimensionality of embeddings


def create_session(graph):
    return tf.compat.v1.Session(graph=graph,
                                config=tf.compat.v1.ConfigProto(
                                        inter_op_parallelism_threads=8))


tf.random.set_seed(5)  # Deterministic results
"""Define Subgraph A"""
graph_a = tf.Graph()
with graph_a.as_default():
    table_a = tf.random.uniform(shape=[N, NDIMS], seed=10)
    ids_a = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_a')
    result_a = tf.nn.embedding_lookup(table_a, ids_a)


def remote_op_a(ids):
    """Mimic a remote op by numpy_function"""
    
    def remote_loopup(ids):
        with create_session(graph_a) as sess:
            return sess.run(result_a, feed_dict={ids_a: ids})
        
    return tf.compat.v1.numpy_function(func=remote_loopup,
                                       inp=[ids],
                                       Tout=tf.float32,
                                       name='remote_op_a')
    

"""Define Subgraph B"""
graph_b = tf.Graph()
with graph_b.as_default():
    ids_b2 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_b2')
    ids_b1 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_b1')
    ids_b1_preprocessed = tf.math.floormod(tf.add(ids_b1, 1), N)
    
    remote_result_a1 = remote_op_a(ids_b1_preprocessed)
    remote_result_a2 = remote_op_a(ids_b2)
    result_b = tf.math.add(remote_result_a1, remote_result_a2 * 2.5)
        
        
def remote_op_b(ids1, ids2):
    """Mimics another remote op"""
    
    def remote_lookup(ids1, ids2):
        with create_session(graph_b) as sess:
            return sess.run(result_b, feed_dict={ids_b1: ids1, ids_b2: ids2})
        
    return tf.compat.v1.numpy_function(func=remote_lookup,
                                       inp=[ids1, ids2],
                                       Tout=tf.float32,
                                       name='remote_op_b')


"""Define Main Graph"""
main_graph = tf.Graph()
with main_graph.as_default():
    ids1 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids1')
    ids2 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids2')
    res_b = remote_op_b(ids1, ids2)
    result = tf.concat([remote_op_a(ids1), res_b], axis=0)
    main_result = tf.reduce_mean(result)



def main():
    with create_session(main_graph) as sess:
        input1 = np.random.uniform(0, N, (10))
        input2 = np.random.uniform(0, N, (10))
        print(sess.run([res_b], feed_dict={ids1: 3, ids2: 3}))
        print(sess.graph.get_operations()[5].inputs)
        
    tf.io.write_graph(graph_a.as_graph_def(), './graphdefs', 'graph_a.pbtxt', as_text=True)
    tf.io.write_graph(graph_b.as_graph_def(), './graphdefs', 'graph_b.pbtxt', as_text=True)
    tf.io.write_graph(main_graph.as_graph_def(), './graphdefs', 'main_graph.pbtxt', as_text=True)

    
if __name__ == "__main__":
    main()
    
    with create_session(graph_b) as sess:
        print(sess.run([remote_result_a1, remote_result_a2], feed_dict={ids_b1: 3, ids_b2: 3}))



































