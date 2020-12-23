import tensorflow as tf

import constants as const
def _parse_layer_params(layer_desc_, num_expected):
    """Extracts parameters from network description layer and raises if there are issues."""
    layer_type_, layer_params = layer_desc_[0], layer_desc_[1:]
    if len(layer_params) != num_expected:
        raise ValueError("Expected {} parameters for layer {} but received {}: {}".format(num_expected, layer_type_, len(layer_params), layer_params))
    return layer_params
def compute_kl_loss(mu_q, ln_var_q, mu_p, ln_var_p):

    ln_det_q = tf.reduce_sum(ln_var_q, axis=[1, 2, 3])
    ln_det_p = tf.reduce_sum(ln_var_p, axis=[1, 2, 3])
    var_p = tf.exp(ln_var_p)
    var_q = tf.exp(ln_var_q)
    tr_qp = tf.reduce_sum(var_q / var_p, axis=[1, 2, 3])
    _, h, w, c = mu_q.get_shape()
    k = float(h.value * w.value * c.value)
    diff = mu_p - mu_q
    term2 = tf.reduce_sum(diff * diff / var_p, axis=[1, 2, 3])
    return 0.5 * (tr_qp + term2 - k + ln_det_p - ln_det_q)

class ConvLstm(tf.keras.Model):
    def __init__(self,  network_description,
                  stochastic=True, weight_decay=0.0, is_training=True, scope='', reuse=False, output_debug=False):
        super(ConvLstm,self).__init__()
        self.scope = scope
        self.network_description = network_description
        if len(network_description) != 1:
            raise Exception("Not Implemented")
        extra = dict()
        layer_desc = network_description[0]
        self.stochastic = stochastic
        self.lstm_size, n_filters, self.number_steps, code_size = _parse_layer_params(layer_desc, 4)
        self.convLSTM_input_c = tf.keras.layers.Conv2D(self.lstm_size * 4,5,1,padding="same",activation = None
                            ,kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),name='convLSTM_input_c')

        self.convLSTM_o_ = tf.keras.layers.Conv2DTranspose(
                            n_filters,
                            4,
                            strides=(4, 4),
                            padding = "same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
                            name='convLSTM_o_')
        self.convLSTM_final_ = tf.keras.layers.Conv2D(
                            n_filters,
                            1,
                            strides=(1, 1),
                            padding = "same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
                            name='convLSTM_final_')

    # @tf.function
    def call(self,inputs,cam_posrot,output_image):
        with tf.name_scope('conv_lstm_full') as scope:
            is_convLSTM_start = False
            out = inputs
            extra = dict()
            for i, layer_desc in enumerate(self.network_description):
                layer_type = layer_desc[0]
                if layer_type == 'convLSTM':

                    if not is_convLSTM_start:
                        '''
                        lstm_size: number of channels used in lstm state/cell
                        n_filters: number of channels in output
                        number_steps: number of lstm steps
                        code_size: #channels in latent code
                        '''
                        input_shape = tf.shape(out)
                        _, h_in, w_in, c_in = out.get_shape()
                        _, h, w, c = output_image.get_shape()
                        sh = int(h)//4
                        sw = int(w)//4

                        if cam_posrot is not None:
                            raise Exception("Not Implemented")
                            # lstm_v = tf.layers.conv2d_transpose(
                            #     cam_posrot,
                            #     12,
                            #     sh,
                            #     strides=(sh, sh),
                            #     use_bias=False,
                            #     padding = "same",
                            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            #     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            #     name=f'convLSTM_v{i}')
                        else:
                            bs = inputs.shape.as_list()[0]
                            lstm_v = tf.zeros((bs, sh, sh, 0))

                            #if int(h_in) != sh: # use convolution to change its size

                        #set to True if input is B x 1 x 1 x C
                        if h_in == 1:
                            raise("Exception Not Implemeted Coz will have to create a seperate keras Model")
                            # lstm_r = tf.layers.conv2d_transpose(
                            #     out,
                            #     int(c_in),
                            #     sh,
                            #     strides=(sh, sh),
                            #     use_bias=False,
                            #     padding = "same",
                            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            #     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            #     name=f'convLSTM_r{i}')
                        else:
                            assert h_in == sh
                            lstm_r = out

                        lstm_h = tf.zeros([input_shape[0], int(h)//4, int(w)//4, self.lstm_size], dtype=tf.float32)
                        lstm_c = tf.zeros([input_shape[0], int(h)//4, int(w)//4, self.lstm_size], dtype=tf.float32)

                        #change this to accomodate a variable channel output
                        dims = 3 + const.EMBEDDING_LOSS * const.embedding_size
                        lstm_u = tf.zeros(output_image.shape.as_list()[:-1]+[dims], dtype=tf.float32)

                        g_mu_var = []
                        lstm_h_g = []
                        lstm_u_g = []
                        for step_id in range(self.number_steps):
                            if step_id > 0: lstm_reuse=True

                            lstm_input = tf.concat([lstm_h, lstm_r, lstm_v], 3)
                            lstm_u_g.append(lstm_u)
                            lstm_h_g.append(lstm_h)
                            if self.stochastic:
                                raise Exception("Not implemeted")
                                # lstm_input_mu_var = tf.layers.conv2d(
                                #     lstm_h,
                                #     code_size * 2,
                                #     5,
                                #     strides=1,
                                #     padding = "same",
                                #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                #     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                #     name=f'convLSTM_mu{i}',
                                #     reuse=lstm_reuse)

                                # lstm_input_mu, lstm_input_log_var = tf.split(lstm_input_mu_var, 2, axis=3)
                                # g_mu_var.append((lstm_input_mu, lstm_input_log_var))
                                # epsilon = tf.random_normal(tf.shape(lstm_input_mu))
                                # lstm_z = lstm_input_mu + tf.exp(0.5 * lstm_input_log_var) * epsilon
                                # lstm_input = tf.concat([lstm_input, lstm_z], 3)


                            lstm_input_all = self.convLSTM_input_c(lstm_input)
                            lstm_input_c, lstm_input_i1, lstm_input_i2, lstm_out = tf.split(lstm_input_all, 4, axis=3)
                            lstm_input_c = tf.keras.activations.sigmoid(lstm_input_c)
                            lstm_input_i1 = tf.keras.activations.sigmoid(lstm_input_i1)
                            lstm_input_i2 = tf.keras.activations.tanh(lstm_input_i2)
                            lstm_out = tf.keras.activations.sigmoid(lstm_out)

                            lstm_input = tf.math.multiply(lstm_input_i1, lstm_input_i2)
                            lstm_c = tf.math.multiply(lstm_c, lstm_input_c) + lstm_input
                            lstm_h = tf.math.multiply(tf.keras.activations.tanh(lstm_c), lstm_out)

                            lstm_final_out = self.convLSTM_o_(lstm_h)

                            lstm_u = lstm_u + lstm_final_out
                        out = lstm_u

                        out = self.convLSTM_final_(out)

                        e_mu_var = []
                        if self.stochastic:
                            raise("Exception")
                        #     lstm_x_q_e = tf.layers.conv2d(
                        #         output_image,
                        #         16,
                        #         5,
                        #         strides=4,
                        #         padding = "same",
                        #         activation=tf.nn.sigmoid,
                        #         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        #         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                        #         name=f'convLSTM_x_q_e_{i}',
                        #         reuse=reuse)
                        #     lstm_h_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)
                        #     lstm_c_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)

                        #     for step_id in range(number_steps):

                        #         lstm_e_reuse = reuse or step_id > 0
                        #         lstm_input_mu_var_e = tf.layers.conv2d(
                        #             lstm_h_e,
                        #             code_size * 2,
                        #             5,
                        #             strides=1,
                        #             padding = "same",
                        #             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        #             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                        #             name=f'convLSTM_mu_e{i}',
                        #             reuse=lstm_e_reuse)
                        #         lstm_input_mu_e, lstm_input_log_var_e = tf.split(lstm_input_mu_var_e, 2, axis=3)

                        #         e_mu_var.append((lstm_input_mu_e, lstm_input_log_var_e))
                        #         if step_id == number_steps - 1:
                        #             break

                        #         lstm_u_e = tf.layers.conv2d(
                        #            lstm_u_g[step_id],
                        #            16,
                        #            5,
                        #            strides=4,
                        #            padding = "same",
                        #            activation=tf.nn.sigmoid,
                        #            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        #            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                        #            name=f'convLSTM_u_e_{i}',
                        #            reuse=lstm_e_reuse)
                        #         lstm_input_e = tf.concat([lstm_h_e, lstm_h_g[step_id], lstm_u_e, lstm_r, lstm_v, lstm_x_q_e], axis=3)

                        #         lstm_input_all = tf.layers.conv2d(
                        #             lstm_input_e,
                        #             code_size * 4,
                        #             5,
                        #             strides=1,
                        #             padding = "same",
                        #             activation=None,
                        #             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        #             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                        #             name=f'convLSTM_input_c_e{i}',
                        #             reuse=lstm_e_reuse)

                        #         lstm_input_c_e, lstm_input_i1_e, lstm_input_i2_e, lstm_out_e = tf.split(lstm_input_all, 4, axis=3)
                        #         lstm_input_c_e = tf.nn.sigmoid(lstm_input_c_e)
                        #         lstm_input_i1_e = tf.nn.sigmoid(lstm_input_i1_e)
                        #         lstm_input_i2_e = tf.nn.tanh(lstm_input_i2_e)
                        #         lstm_out_e = tf.nn.sigmoid(lstm_out_e)

                        #         lstm_input_e = tf.multiply(lstm_input_i1_e, lstm_input_i2_e)
                        #         lstm_c_e = tf.multiply(lstm_c_e, lstm_input_c_e) + lstm_input_e
                        #         lstm_h_e = tf.multiply(tf.nn.tanh(lstm_c_e), lstm_out_e)

                        kl_loss = 0
                        for layer_id, g_mu_var_ in enumerate(g_mu_var):
                            kl_loss += tf.math.reduce_mean(compute_kl_loss(
                                e_mu_var[layer_id][0],
                                e_mu_var[layer_id][1],
                                g_mu_var_[0],
                                g_mu_var_[1]
                            ))

                        if isinstance(kl_loss, int):
                            kl_loss = tf.constant(0.0, dtype = tf.float32)
                            
                        extra['kl_loss'] = kl_loss
                else:
                    raise ValueError("Unknown layer type '{}' with params {}".format(layer_type, layer_desc[1:]))


        return out, extra
if __name__ == "__main__":
    cl = ConvLstm([['convLSTM', 128, 35, 4, 128]], stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,weight_decay = 1E-5,is_training = const.mode == 'train',reuse = False,output_debug = False)
    val = cl(tf.zeros([2,16,16,256]),None,tf.zeros([2,64,64,3]))
    print(val)

