import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from .domain_norm import DomainNorm
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.layers import Layer, Dropout

from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.engine.topology import Layer

class MLoRAFCN(Layer):

    def __init__(self,
                 n_domain,
                 units,

                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 lora_r=4,
                 lora_reduce=-1,
                 dropout_rate=0.5,
                 is_finetune=False,

                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MLoRAFCN, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.n_domain = n_domain

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        print("self.kernel_initializer: ", self.kernel_initializer)
        print("self.bias_initializer: ", self.bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.supports_masking = False
        self.lora_r = lora_r
        if lora_r < 1 and lora_reduce >= 1:
            self.lora_r = max(int(units/lora_reduce), 1)

        self.is_finetune = tf.constant(1.0 if is_finetune else 0.0, dtype=tf.float32)
        # Attention部分的参数
        self.nb_head = 1  # 多头注意力机制中的头数
        self.size_per_head = 16  # 每个头的大小
        self.output_dim = self.nb_head * self.size_per_head  # 输出维度

    def build(self, input_shape):
        input_shape, domain_indicator_shape = input_shape
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = [InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value}),
                           InputSpec(shape=domain_indicator_shape)]

        # Domain
        self.a_kernel = self.add_weight(
            "A_Kernel",
            shape=[self.n_domain, input_shape[-1].value, self.lora_r],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)


        print("self.a_kernel", self.a_kernel)
        # self.a_kernel1 = self.is_finetune * self.a_kernel1 + (1.0-self.is_finetune) * tf.stop_gradient(self.a_kernel1)


        self.b_kernel = self.add_weight(
            "B_Kernel",
            shape=[self.n_domain, self.lora_r, self.units],
            initializer=self.bias_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        print("self.b_kernel", self.b_kernel)
        # self.b_kernel1 = self.is_finetune * self.b_kernel1 + (1.0-self.is_finetune) * tf.stop_gradient(self.b_kernel1)


        if self.use_bias:
            self.domain_bias = self.add_weight(
                "domain_bias",
                shape=[self.n_domain, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            print("self.domain_bias", self.domain_bias)
            # self.domain_bias1 = self.is_finetune * self.domain_bias1 + (1.0 - self.is_finetune) * tf.stop_gradient(self.domain_bias1)
        else:
            self.domain_bias = None

        # MLP weight and bias
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.kernel", self.kernel)
        # self.domain_bias = self.is_finetune * tf.stop_gradient(self.kernel) + (1.0 - self.is_finetune) * self.kernel

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            print("self.bias", self.bias)
            # self.bias = self.is_finetune * tf.stop_gradient(self.bias) + (1.0 - self.is_finetune) * self.bias
        else:
            self.bias = None

        # Attention
        #维度 r*dk
        self.WQ = self.add_weight(name='WQ',
                                  shape=(self.lora_r, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)  # 查询的权重矩阵
        self.WK = self.add_weight(name='WK',
                                  shape=(self.lora_r, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)  # 键的权重矩阵
        self.WV = self.add_weight(name='WV',
                                  shape=(self.lora_r, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)  # 值的权重矩阵
        self.linear = self.add_weight(name='linear',
                                      shape=(self.output_dim , self.lora_r),
                                      initializer='glorot_uniform',
                                      trainable=True) # 线性层
        ####分界线
        #B与主干比例因子
        self.factor = self.add_weight(name='factor',
                                      shape=(1 , self.n_domain),
                                      initializer='glorot_uniform',
                                      trainable=True) # 线性层
        ####

        self.dropout_layers = Dropout(self.dropout_rate)

        self.built = True

    def Attention(self, x):

        Q_seq, K_seq, V_seq = x

        # 对Q、K、V做线性变换 (?,10,dk)
        Q_seq = K.dot(Q_seq, self.WQ)  # 计算查询
        # Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))  # 重塑查询矩阵
        # Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))  # 转置查询矩阵
        K_seq = K.dot(K_seq, self.WK)  # 计算键
        # K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))  # 重塑键矩阵
        # K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))  # 转置键矩阵
        V_seq = K.dot(V_seq, self.WV)  # 计算值
        # V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))  # 重塑值矩阵
        # V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))  # 转置值矩阵

        # 计算内积，然后softmax （？，10，10）
        A = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / self.size_per_head**0.5  # 计算注意力分数
        A = K.softmax(A)  # 计算softmax

        # 输出attention(?,10,16)
        O_seq = K.batch_dot(A, V_seq, axes=[2, 3])  # 计算输出
        # O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))  # 转置输出矩阵
        # O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))  # 重塑输出矩阵
        #维度与输入一致 linear层 self.output_dim *lora_r
        O_seq = K.dot(O_seq, self.linear)
        #domain数量*lora_r
        return O_seq

    def call(self, inputs, training=None, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        #MLP
        outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        outputs = self.dropout_layers(outputs, training=training)

        # domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        # 这里使用了对应domain id的A和B
        # A的部分改成提取全部的a_kernel
        # domain_a_kernel = nn.embedding_lookup(self.a_kernel, idx)
        #包含所有的A 注意索引 domain_a_kernels[1] 对应 domain2_a_kernel
        domain_a_kernels = [nn.embedding_lookup(self.a_kernel, i) for i in range(1, 11)]

        #仅 transformer
        # domain_b_kernel = nn.embedding_lookup(self.b_kernel, idx)

        #####分界线
        #B和主干 所有的B
        domain_b_kernels = [nn.embedding_lookup(self.b_kernel, i) for i in range(1, 11)]
        ####

        if self.use_bias:
            domain_bias = nn.embedding_lookup(self.domain_bias, idx)
        #加transformer
        #先把输入过每一个A，然后组合起来
        #所有过A之后的  domainA_outputs[1] 对应 domainA2_outputs
        domainA_outputs = [gen_math_ops.mat_mul(inputs, domain_a_kernels[i]) for i in range(10)]
        #组合起来 为transformer的input (?,10,r)
        combined_matrix = tf.stack(domainA_outputs, axis=1)
        #transformer output(?,10,r)
        O_seq = self.Attention([combined_matrix,combined_matrix,combined_matrix])
        # 每一行加起来(?,1,r)
        average_row = tf.reduce_sum(O_seq, axis=1, keepdims=True)
        # (?,r)
        domain_outputs = tf.squeeze(average_row,axis=1)
        # domain_outputs = gen_math_ops.mat_mul(inputs, domain_a_kernel)

        #仅 transformer
        # 然后过B
        # domain_outputs = gen_math_ops.mat_mul(domain_outputs, domain_b_kernel)

        #####分界线
        # B和主干
        domainB_outputs= [gen_math_ops.mat_mul(domain_outputs, domain_b_kernels[i]) for i in range(10)]

        combined_matrixB = tf.stack(domainB_outputs, axis=1)

        softmax_factors = tf.nn.softmax(self.factor, axis=-1)
        expanded_factors = tf.expand_dims(softmax_factors, axis=-1)
        scaled_inputs = combined_matrixB * expanded_factors
        sum = tf.reduce_sum(scaled_inputs, axis=1, keepdims=True)
        domain_outputs = tf.squeeze(sum, axis=1)
        #####



        #domain 的bias 在softmax之后再加
        if self.use_bias:
            domain_outputs = nn.bias_add(domain_outputs, domain_bias)
        # domain_outputs1 = DomainNorm(self.n_domain_1)([domain_outputs1, domain_indicator1])
        outputs += domain_outputs


        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            "lora_r": self.lora_r,
            'n_domain': self.n_domain,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MLoRAFCN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))