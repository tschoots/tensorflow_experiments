import tensorflow as tf


def main():
    print("hallo")

    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly because of default

    print(node1, node2)

    sess = tf.Session()
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print("node 3 : ", node3)
    print("sess.run(node3) : ", sess.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a,b)

    print(sess.run(adder_node, {a: 3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b:[2,4]}))

    add_and_tripple = adder_node * 3
    print(sess.run(add_and_tripple, {a: 3, b:4.5}))

    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(linear_model, {x:[1,2,3,4]}))

    # loss function
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print("loss with initial values: " , sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))

    # manualy assign the optimal values , bit silly ;-(
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print("loss with manual optimum : " ,  sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # now real training stuff
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    iterations = 1000
    sess.run(init) # reset values to inccorrect defaults
    for i in range(iterations):
        sess.run(train, {x:[1,2,3,4], y:[0,-1, -2, -3]})

    print("%d iterations : %s" %(iterations, sess.run([W,b])))

    iterations = 10000
    sess.run(init)  # reset values to inccorrect defaults
    for i in range(iterations):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print("%d iterations : %s" % (iterations, sess.run([W, b])))






if __name__ == "__main__":
    main()