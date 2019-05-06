
from torch import nn




class RNNLM(nn.Module):

    def __init__(self):
        super(RNNLM, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)


def main():
    model = RNNLM()
    model.train(iters=3)

if __name__ == "__main__":
    main()