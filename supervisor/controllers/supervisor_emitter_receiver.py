from abc import abstractmethod
from collections.abc import Iterable

from .supervisor_env import SupervisorEnv


class SupervisorEmitterReceiver(SupervisorEnv):
    def __init__(self,
                 emitter_name=["emitter"],
                 receiver_name=["receiver"],
                 time_step=None):

        super(SupervisorEmitterReceiver, self).__init__()

        if time_step is None:
            self.timestep = int(self.supervisor.getBasicTimeStep())
        else:
            self.timestep = time_step

        self.emitter = None
        self.receiver = None
        self.initialize_comms(emitter_name, receiver_name)

    def initialize_comms(self, emitter_name, receiver_name):
        self.emitter = [self.supervisor.getEmitter(emitter_name[i]) for i in range(len(emitter_name))]
        self.receiver = [self.supervisor.getReceiver(receiver_name[i]) for i in range(len(emitter_name))]
        for i in range(len(self.receiver)):
            self.receiver[i].enable(self.timestep)
        return self.emitter, self.receiver

    def step(self, action):
        if self.supervisor.step(self.timestep) == -1:
            exit()

        self.handle_emitter(action)
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    @abstractmethod
    def handle_emitter(self, action):
        pass

    @abstractmethod
    def handle_receiver(self):
        pass

    def get_timestep(self):
        return self.timestep


class SupervisorCSV(SupervisorEmitterReceiver):
    def __init__(self,
                 emitter_name=["emitter"],
                 receiver_name=["receiver"],
                 time_step=None):
        super(SupervisorCSV, self).__init__(emitter_name, receiver_name,
                                            time_step)

    def handle_emitter(self, action):
        assert isinstance(action, Iterable), \
            "The action object should be Iterable"
        for i in range(len(self.emitter)):
            message = (",".join(map(str, action[i]))).encode("utf-8")
            self.emitter[i].send(message)


    def handle_receiver(self):
        string_message = [[None] for i in range(len(self.receiver))]
        for i in range(len(self.receiver)):
            if self.receiver[i].getQueueLength() > 0:
                string_message[i] = self.receiver[i].getData().decode("utf-8").split(",")
                self.receiver[i].nextPacket()

        return string_message