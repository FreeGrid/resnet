import tensorflow as tf

cluster = tf.train.ClusterSpec({
    'node1':[
        '192.168.136.101:2222'
    ],
	'node2':[
		'192.168.136.102:2222'
	]
})

server = tf.train.Server(cluster,job_name='node2',task_index=0)
server.join()
