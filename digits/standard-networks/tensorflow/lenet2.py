from model import Tower
from utils import model_property

class UserModel(Tower):
    
  def __init__(self):
    super(UserModel, self).__init__('lenet5', 28, 32, 0.005)

    def inference(self, cnn):
        # Note: This matches TF's MNIST tutorial model
        cnn.conv(32, 5, 5)
        cnn.mpool(2, 2)
        cnn.conv(64, 5, 5)
        cnn.mpool(2, 2)
        cnn.reshape([-1, 64 * 7 * 7])
        cnn.affine(512)
