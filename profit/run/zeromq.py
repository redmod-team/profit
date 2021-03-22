""" zeromq Interface

communication via 0MQ
 - router + req for data transfer
 - router + rep for heartbeat
  - only one router is used, the worker maintains two sockets with two different addresses

internal storage with a structured numpy array
 - not saved to disk

no backup capability

Ideas & Help from the 0MQ Guide (zguide.zeromq.org, examples are MIT licence)
"""

from .runner import RunnerInterface
from .worker import Interface

import zmq
import numpy as np
import json


@RunnerInterface.register('zeromq')
class ZeroMQRunnerInterface(RunnerInterface):
    def __init__(self, *args):
        if 'FLAGS' not in [var[0] for var in self.internal_vars]:
            self.internal_vars += [('FLAGS', np.bool8)]
        super().__init__(*args)
        self.socket = zmq.Context.instance().socket(zmq.ROUTER)
        self.socket.bind(self.config['bind'])
        self.logger.info('connected')

    @property
    def data(self):
        return self._data

    def handle_msg(self, address: bytes, msg: list):
        #if address[:3] == b'rep':  # Heartbeat
        #    pass # ToDo
        if address[:3] == b'req':  # run123
            run_id = int(address[3:])
            self.logger.debug(f'received message {address}: {msg}')
            if msg[0] == b'READY':
                input_descr = json.dumps(self.input_vars).encode()
                input_data = self.data[run_id][[t[0] for t in self.input_vars]]
                output_descr = json.dumps(self.output_vars).encode()
                self.socket.send_multipart([address, b'', input_descr, input_data, output_descr])
                self.data['FLAGS'][run_id] |= 0x02
            elif msg[0] == b'DATA':
                self.data[[t[0] for t in self.output_vars]][run_id] = np.frombuffer(msg[1], dtype=self.output_vars)
                self.data['DONE'][run_id] = True
                self.data['FLAGS'][run_id] |= 0x08
                self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'TIME':
                self.data['TIME'][run_id] = msg[1]
                self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'DIE':
                self.data['FLAGS'][run_id] |= 0x04
                # no acknowledgement needed, DIE is never retried
            else:
                self.logger.warning(f'received unknown message {address}: {msg}')
        else:
            self.logger.warning(f'received message from unknown client {address}: {msg}')

    def cancel(self, run_id):
        if not self.data['FLAGS'][run_id] & 0b11100:
            if self.data['FLAGS'][run_id] & 0b10:
                self.socket.send_multipart([b'rep' + bytes([run_id]), b'', b'CANCEL'])
                self.data['FLAGS'][run_id] |= 0x10  # cancelled
                return True
        return False  # not possible

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: zeromq
        bind: tcp://*:9000
        connect: tcp://localhost:9000
        """
        if 'bind' not in config:
            config['bind'] = 'tcp://*:9000'
        if 'connect' not in config:
            config['connect'] = 'tcp://localhost:9000'


@Interface.register('zeromq')
class ZeroMQInterface(Interface):
    REQUEST_TIMEOUT = 2500  # ms
    REQUEST_TRIES = 3

    def __init__(self, *args):
        super().__init__(*args)
        self.connect()  # self.req, self.rep
        self._done = False
        self._time = 0
        self.request('READY')  # self.input, self.output

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value: int):
        self._time = value
        self.request('TIME')

    def done(self):
        self.request('DATA')

    def connect(self, who=None):
        if who is None:
            self.connect('req')
            self.connect('rep')
            return self.req, self.rep
        elif who == 'req':
            self.req = zmq.Context.instance().socket(zmq.REQ)
            self.req.setsockopt(zmq.IDENTITY, b'req' + bytes([self.worker.run_id]))
            self.req.connect(self.config['connect'])
            self.logger.info('REQ connected')
        elif who == 'rep':
            self.rep = zmq.Context.instance().socket(zmq.REP)
            self.rep.setsockopt(zmq.IDENTITY, b'rep' + bytes([self.worker.run_id]))
            self.req.connect(self.config['connect'])
            self.logger.info('REP connected')
        else:
            raise ValueError

    def request(self, request):
        """ 0MQ - Lazy Pirate Pattern """
        if request not in ['READY', 'DATA', 'TIME']:
            raise ValueError('unknown request')
        tries = self.REQUEST_TRIES
        while True:
            msg = [request.encode()]
            if request == 'DATA':
                msg.append(self.output)
            elif request == 'TIME':
                msg.append(np.uint(self.time))
            self.req.send_multipart(msg)
            if self.req.poll(self.REQUEST_TIMEOUT) & zmq.POLLIN:
                try:
                    response = self.req.recv_multipart()
                    self.logger.debug(f'REQ {request}: received {response}')
                    if request == 'READY':
                        input_descr, input_data, output_descr = response
                        input_descr = [tuple(column) for column in json.loads(input_descr.decode())]
                        output_descr = [tuple(column) for column in json.loads(output_descr.decode())]
                        self.input = np.frombuffer(input_data, dtype=input_descr)[0]
                        self.output = np.zeros(1, dtype=output_descr)[0]
                        self.logger.info('REQ READY: received input data')
                        return
                    else:
                        assert response[0] == b'ACK'
                        self.logger.info(f'REQ {request}: message acknowledged')
                        return
                except (ValueError, AssertionError):
                    self.logger.error(f'REQ {request}: malformed reply')
            else:
                self.logger.warning(f'REQ {request}: no response')
                tries += 1

            if tries >= self.REQUEST_TRIES:
                self.logger.error(f'REQ {request}: {tries} requests unsuccessful, abandoning')
                self.worker.cancel()  # TODO
                return

            # close and reopen the socket
            self.req.close(linger=0)
            self.connect('req')





