class Sequential(object):

  def __init__(self, inputs):
    self._output = inputs

  def add(self, layer):

    layer.apply_input(self.output)
    self._output = layer()

  @property
  def output(self):
    return self._output
