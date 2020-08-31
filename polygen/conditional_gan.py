#conditional on imput image


class ImageToVertexModel(VertexModel):
  """Generative model of quantized mesh vertices with image conditioning.

  Operates on flattened vertex sequences with a stopping token:

  [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

  Input vertex coordinates are embedded and tagged with learned coordinate and
  position indicators. A transformer decoder outputs logits for a quantized
  vertex distribution. Image inputs are encoded and used to condition the
  vertex decoder.
  """

  def __init__(self,
               res_net_config,
               decoder_config,
               quantization_bits,
               use_discrete_embeddings=True,
               max_num_input_verts=2500,
               name='image_to_vertex_model'):
    """Initializes VoxelToVertexModel.

    Args:
      res_net_config: Dictionary with ResNet config.
      decoder_config: Dictionary with TransformerDecoder config.
      quantization_bits: Number of quantization used in mesh preprocessing.
      use_discrete_embeddings: If True, use discrete rather than continuous
        vertex embeddings.
      max_num_input_verts: Maximum number of vertices. Used for learned position
        embeddings.
      name: Name of variable scope
    """
    super(ImageToVertexModel, self).__init__(
        decoder_config=decoder_config,
        quantization_bits=quantization_bits,
        max_num_input_verts=max_num_input_verts,
        use_discrete_embeddings=use_discrete_embeddings,
        name=name)

    with self._enter_variable_scope():
      self.res_net = ResNet(num_dims=2, **res_net_config)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):

    # Pass images through encoder
    image_embeddings = self.res_net(
        context['image'] - 0.5, is_training=is_training)

    # Add 2D coordinate grid embedding
    processed_image_resolution = tf.shape(image_embeddings)[1]
    x = tf.linspace(-1., 1., processed_image_resolution)
    image_coords = tf.stack(tf.meshgrid(x, x), axis=-1)
    image_coord_embeddings = tf.layers.dense(
        image_coords,
        self.embedding_dim,
        use_bias=True,
        name='image_coord_embeddings')
    image_embeddings += image_coord_embeddings[None]

    # Reshape spatial grid to sequence
    batch_size = tf.shape(image_embeddings)[0]
    sequential_context_embedding = tf.reshape(
        image_embeddings, [batch_size, -1, self.embedding_dim])

    return None, sequential_context_embedding

