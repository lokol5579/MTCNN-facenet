from keras import backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, \
    Concatenate, Conv2D, Add, Activation, Lambda

''' 通道注意力机制：
    对输入feature map进行spatial维度压缩时，作者不单单考虑了average pooling，
    额外引入max pooling作为补充，通过两个pooling函数以后总共可以得到两个一维矢量。
    global average pooling对feature map上的每一个像素点都有反馈，而global max pooling
    在进行梯度反向传播计算只有feature map中响应最大的地方有梯度的反馈，能作为GAP的一个补充。
'''


def channel_attention(input_feature, ratio=8, layer_num=0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = int(input_feature.shape[channel_axis])

    # shared_layer_one = Dense(channel // ratio, kernel_initializer='he_normal', activation='relu', use_bias=True, bias_initializer='zeros', name="shared_layer_one_%d"%layer_num)
    #
    # shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name="shared_layer_two_%d"%layer_num)

    avg_pool = GlobalAveragePooling2D(name="global_average_pooling_channel_%d"%layer_num)(input_feature)
    avg_pool = Reshape((1, 1, channel), name="reshape_channel_%d"%layer_num)(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = Dense(channel // ratio, kernel_initializer='he_normal', activation='relu', use_bias=True, bias_initializer='zeros', name="shared_layer_one_%d"%layer_num)(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name="shared_layer_two_%d"%layer_num)(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    layer_num = layer_num + 1

    max_pool = GlobalMaxPooling2D(name="global_max_pooling_channel_%d"%layer_num)(input_feature)
    max_pool = Reshape((1, 1, channel), name="reshape_channel_%d"%layer_num)(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = Dense(channel // ratio, kernel_initializer='he_normal', activation='relu', use_bias=True, bias_initializer='zeros', name="shared_layer_one_%d"%layer_num)(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name="shared_layer_two_%d"%layer_num)(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    layer_num = layer_num + 1

    cbam_feature = Add(name="add_channel_%d"%layer_num)([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid', name="activation_channel_%d"%layer_num)(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2), name="permute_channel_%d"%layer_num)(cbam_feature)

    return multiply([input_feature, cbam_feature])


''' 空间注意力机制:
    还是使用average pooling和max pooling对输入feature map进行压缩操作，
    只不过这里的压缩变成了通道层面上的压缩，对输入特征分别在通道维度上做了
    mean和max操作。最后得到了两个二维的feature，将其按通道维度拼接在一起
    得到一个通道数为2的feature map，之后使用一个包含单个卷积核的隐藏层对
    其进行卷积操作，要保证最后得到的feature在spatial维度上与输入的feature map一致，
'''


def spatial_attention(input_feature, layer_num=0):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1), name="permute_spatial_%d"%layer_num)(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name="avg_lambda_spatial_%d"%layer_num)(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True), name="max_lambda_spatial_%d"%layer_num)(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3, name="concatenate_spatial_%d"%layer_num)([avg_pool, max_pool])
    assert concat.shape[-1] == 2

    layer_num = layer_num + 1

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False, name="conv_spatial_%d"%layer_num)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2), name="permute_spatial_%d"%layer_num)(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, layer_num=0):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    # 实验验证先通道后空间的方式比先空间后通道或者通道空间并行的方式效果更佳
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)

    return cbam_feature