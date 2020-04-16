from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """All of the policy classes must inherit
    :class:`BaseAgent`.

    A policy class typically has four parts:

    * :meth:`BaseAgent.__init__`: initialize the policy, \
        including coping the target network and so on;
    * :meth:`BaseAgent.forward`: compute action with given \
        observation;
    * :meth:`BaseAgent.process_fn`: pre-process data from \
        the replay buffer (this function can interact with replay buffer);
    * :meth:`BaseAgent.learn`: update policy with a given \
        batch of data.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation ``obs`` (may be a ``numpy.ndarray`` or \
        ``torch.Tensor``), hidden state ``state`` (for RNN usage), and other \
        information ``info`` provided by the environment.
    2. Output: some ``logits`` and the next hidden state ``state``. The logits\
        could be a tuple instead of a ``torch.Tensor``. It depends on how the \
        policy process the network output. For example, in PPO, the return of \
        the network might be ``(mu, sigma), state`` for Gaussian policy.

    Since :class:`~tianshou.policy.BaseAgent` inherits ``torch.nn.Module``,
    you can use :class:`~tianshou.policy.BaseAgent` almost the same as
    ``torch.nn.Module``, for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), 'policy.pth')
        policy.load_state_dict(torch.load('policy.pth'))
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def learn(self, batch, **kwargs):
        """Update policy with a given batch of data.

        :return: A dict which includes loss and its corresponding label.
        """
        pass

    @abstractmethod
    def get_action(self, state):
        """
        Compute action given a state
        :param state:
        :return: action
        """

        pass
