import tensorflow as tf
import numpy as np


def i_basis(nbit, NORM_PPF_0_75=0.6745):
    init_basis = [(NORM_PPF_0_75 * 2 / (2 ** nbit - 1)) * (2. ** i) for i in range(nbit)]
    init_basis = np.array(init_basis).astype(np.float32)
    init_basis = init_basis[...,np.newaxis]

    return init_basis


def i_basis_xy(nbit, xmax, xmin, NORM_PPF_0_75=0.6745):
    eps = 1e-5
    xybasis = np.zeros([nbit])
    if xmin>=eps:
        tmp_basis = [(NORM_PPF_0_75 * 2 / (2 ** (nbit-1) - 1)) * (2. ** i) for i in range(nbit-1)]
        tmp_basis = np.array(tmp_basis)
        tmp_basis = tmp_basis*(xmax-xmin)
        xybasis[:nbit-1] = tmp_basis
        xybasis[nbit-1] = xmin
        xybasis = np.sort(xybasis)
    else:
        tmp_basis = [(NORM_PPF_0_75 * 2 / (2 ** nbit - 1)) * (2. ** i) for i in range(nbit)]
        tmp_basis = np.array(tmp_basis)
        tmp_basis = tmp_basis*(xmax-xmin)
        xybasis = tmp_basis

    xybasis = xybasis[...,np.newaxis]
    xybasis = np.array(xybasis).astype(np.float32)

    return xybasis


def np_inv(ain):
    b = np.zeros(ain.shape)

    try:
        b = np.linalg.inv(ain)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        print('Not invertible')
        pass
    #else:
        # continue with what you were doing
        #print('\nRun else')
        
    return b


def New_basis(Bits_y_in, X_in, nbit, basis_in):
    nBT = np.transpose(Bits_y_in)
    #print('nBT.shape: ' + str(nBT.shape))
    # calculate BTxB
    nBTxB = []
    for i in range(nbit):
        for j in range(nbit):
            nBTxBij = nBT[i]*nBT[j]
            nBTxBij = np.sum(nBTxBij)
            nBTxB.append(nBTxBij)
    nBTxB = np.reshape(np.array(nBTxB), [nbit, nbit])
    #print('\nnBTxB:\n' + str(nBTxB))
    
    # calculate BTxX
    nreshape_x = np.reshape(X_in, [-1])
    nBTxX = []
    for i in range(nbit):
        nBTxXi0 = nBT[i]*nreshape_x
        nBTxXi0 = np.sum(nBTxXi0)
        nBTxX.append(nBTxXi0)
    nBTxX = np.reshape(np.array(nBTxX), [nbit, 1])

    #print('\nnBTxX:\n' + str(nBTxX))
    
    a = nBTxB.copy()
    eps = 1e-7
    for i in range(nbit):
        if a[i,0]<eps:
            a[i,:]=0.0; a[:,i]=0.0; a[i,i]=1.0
            
    b = np_inv(a)
    #print('\nb:\n' + str(b))

    bx = nBTxX.copy()
    for i in range(nbit):
        if nBTxB[i,0]<eps:bx[i]=0.0
        
    nnew_basis = np.matmul(b, bx)  # calculate new basis
    nnew_basis[np.abs(nnew_basis)<eps] = basis_in[np.abs(nnew_basis)<eps]
    #print('nnew_basis: \n' + str(nnew_basis))

    return nnew_basis


def get_placeholder(nx, ny, nbit):
    x = tf.placeholder(tf.float32,[nx,ny])
    basis = tf.placeholder(tf.float32,[nbit,1])

    return x, basis


def get_placeholder_4D(nx, ny, NC, nbit):
    x = tf.placeholder(tf.float32,[None,nx,ny,NC])
    basis = tf.placeholder(tf.float32,[nbit,1])

    return x, basis


def get_multipliers(nbit):
    bit_dims = [nbit, 1]
    num_levels = 2 ** nbit
    # initialize level multiplier
    init_level_multiplier = []
    for i in range(0, num_levels):
        level_multiplier_i = [0. for j in range(nbit)]
        level_number = i
        for j in range(nbit):
            level_multiplier_i[j] = float(level_number % 2)
            level_number = level_number // 2
        init_level_multiplier.append(level_multiplier_i)
    

    # initialize threshold multiplier
    init_thrs_multiplier = []
    for i in range(1, num_levels):
        thrs_multiplier_i = [0. for j in range(num_levels)]
        thrs_multiplier_i[i - 1] = 0.5
        thrs_multiplier_i[i] = 0.5
        init_thrs_multiplier.append(thrs_multiplier_i)

    return init_level_multiplier, init_thrs_multiplier


def q_run(x, basis, nbit, flag_train, init_level_multiplier, init_thrs_multiplier):
    num_levels = 2 ** nbit
    # calculate levels and sort
    level_codes = tf.constant(init_level_multiplier)
    levels = tf.matmul(level_codes, basis)
    levels, sort_id = tf.nn.top_k(tf.transpose(levels, [1, 0]), num_levels)
    levels = tf.reverse(levels, [-1])
    sort_id = tf.reverse(sort_id, [-1])
    levels = tf.transpose(levels, [1, 0])
    sort_id = tf.transpose(sort_id, [1, 0])
    # calculate threshold
    thrs_multiplier = tf.constant(init_thrs_multiplier)
    thrs = tf.matmul(thrs_multiplier, levels)
    # calculate output y and its binary code
    y = tf.zeros_like(x)  # output
    reshape_x = tf.reshape(x, [-1])
    zero_dims = tf.stack([tf.shape(reshape_x)[0], nbit])
    bits_y = tf.fill(zero_dims, 0.)
    zero_y = tf.zeros_like(x)
    zero_bits_y = tf.fill(zero_dims, 0.)
    for i in range(num_levels - 1):
        g = tf.greater(x, thrs[i])
        y = tf.where(g, zero_y + levels[i + 1], y)
        bits_y = tf.where(tf.reshape(g, [-1]), zero_bits_y + level_codes[sort_id[i + 1][0]], bits_y)

    # necessary for training
    new_basis = 0
    if flag_train ==1:
        BT = tf.matrix_transpose(bits_y)
        # calculate BTxB
        BTxB = []
        for i in range(nbit):
            for j in range(nbit):
                BTxBij = tf.multiply(BT[i], BT[j])
                BTxBij = tf.reduce_sum(BTxBij)
                BTxB.append(BTxBij)
        BTxB = tf.reshape(tf.stack(values=BTxB), [nbit, nbit])
        BTxB_inv = tf.matrix_inverse(BTxB)
        # calculate BTxX
        BTxX = []
        for i in range(nbit):
            BTxXi0 = tf.multiply(BT[i], reshape_x)
            BTxXi0 = tf.reduce_sum(BTxXi0)
            BTxX.append(BTxXi0)
        BTxX = tf.reshape(tf.stack(values=BTxX), [nbit, 1])

        new_basis = tf.matmul(BTxB_inv, BTxX)  # calculate new basis


    x_clip = tf.minimum(x, levels[num_levels - 1])  # gradient clip
    y_clip = x_clip + tf.stop_gradient(-x_clip) + tf.stop_gradient(y)  # gradient: y=clip(x)
    
    return y_clip, bits_y, new_basis












