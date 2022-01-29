""" zeromq Interface

Ideas & Help from the 0MQ Guide (zguide.zeromq.org, examples are licensed with MIT)
"""

from .runner import RunnerInterface
from .worker import Interface

import zmq
import numpy as np
import json
from time import sleep
from logging import Logger
import os


@RunnerInterface.register('zeromq')
class ZeroMQRunnerInterface(RunnerInterface):
    """ Runner-Worker Interface using the lightweight message queue `ZeroMQ <https://zeromq.org/>`_

    - can use different transport systems, most commonly tcp
    - can be used efficiently on a cluster (tested)
    - expected to be inefficient for a large number of small, locally run simulations where communication overhead is
      a concern (unverified, could be mitigated by using a different transport system)
    - known issue: some workers were unable to establish a connection with three tries, reason unknown
    """
    def __init__(self, config, size, input_config, output_config, *, logger_parent: Logger = None):
        if 'FLAGS' not in [var[0] for var in self.internal_vars]:
            self.internal_vars += [('FLAGS', np.byte.__name__)]
        super().__init__(config, size, input_config, output_config, logger_parent=logger_parent)
        self.socket = zmq.Context.instance().socket(zmq.ROUTER)
        if self.config['address'] is None:
            bind = f'{self.config["transport"]}://*:{self.config["port"]}'
        else:
            bind = self.config['bind']
        self.socket.bind(bind)
        self.logger.info(f'connected to {bind}')

    def poll(self):
        self.logger.debug('polling: checking for messages')
        while self.socket.poll(timeout=10, flags=zmq.POLLIN):
            msg = self.socket.recv_multipart()
            # ToDo: Heartbeats
            self.handle_msg(msg[0], msg[2:])

    def handle_msg(self, address: bytes, msg: list):
        if address[:4] == b'req_':  # req_123
            run_id = int(address[4:])
            self.logger.debug(f'received {msg[0]} from run {run_id}')
            if msg[0] == b'READY':
                input_descr = json.dumps(self.input_vars).encode()
                output_descr = json.dumps(self.output_vars).encode()
                self.logger.debug(f'send input {input_descr} + {self.input[run_id]} + output {output_descr}')
                self.socket.send_multipart([address, b'', input_descr, self.input[run_id], output_descr])
                self.internal['FLAGS'][run_id] |= 0x02
            elif msg[0] == b'DATA':
                self.output[run_id] = np.frombuffer(msg[1], dtype=self.output_vars)
                self.logger.debug(f'received output {np.frombuffer(msg[1], dtype=self.output_vars)}')
                self.internal['DONE'][run_id] = True
                self.internal['FLAGS'][run_id] |= 0x08
                self.logger.debug('acknowledge DATA')
                self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'TIME':
                self.internal['TIME'][run_id] = np.frombuffer(msg[1], dtype=np.uint)
                self.logger.debug('acknowledge TIME')
                self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'DIE':
                self.internal['FLAGS'][run_id] |= 0x04
                self.logger.debug('acknowledge DIE')
                self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            else:
                self.logger.warning(f'received unknown message {address}: {msg}')
        else:
            self.logger.warning(f'received message from unknown client {address}: {msg}')

    def clean(self):
        self.logger.debug('cleaning: closing socket')
        self.socket.close(0)


@Interface.register('zeromq')
class ZeroMQInterface(Interface):
    """ Runner-Worker Interface using the lightweight message queue `ZeroMQ <https://zeromq.org/>`_

    counterpart to :py:class:`ZeroMQRunnerInterface`
    """
    def __init__(self, config, run_id: int, *, logger_parent: Logger = None):
        super().__init__(config, run_id, logger_parent=logger_parent)
        self.connect()  # self.socket
        self._done = False
        self._time = 0
        self.request('READY')  # self.input, self.output

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value: int):
        self._time = value
        try:
            self.request('TIME')
        except ConnectionError:
            self.logger.info('saving time to "profit_time.txt"')
            with open('profit_time.txt', 'w') as file:
                file.write(str(value))

    def done(self):
        try:
            self.request('DATA')
        except ConnectionError:
            self.logger.info('saving data to "profit_results.npy"')
            np.save('profit_results.npy', self.output)
            self.socket.close(0)
            # we want the Worker to exit with an error and don't perform cleanup
            raise
        self.socket.close(0)

    def connect(self):
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, f'req_{self.run_id}'.encode())
        if self.config['connect'] is None:
            address = os.environ.get('PROFIT_RUNNER_ADDRESS') or 'localhost'
            connect = f'{self.config["transport"]}://{address}:{self.config["port"]}'
        else:
            connect = self.config['connect']
        self.socket.connect(connect)
        self.logger.info(f'connected to {connect}')

    def request(self, request):
        """ 0MQ - Lazy Pirate Pattern """
        if request not in ['READY', 'DATA', 'TIME']:
            raise ValueError('unknown request')
        tries = 0
        while True:
            msg = [request.encode()]
            if request == 'DATA':
                msg.append(self.output)
            elif request == 'TIME':
                msg.append(np.uint(self.time))
            self.socket.send_multipart(msg)
            if self.socket.poll(timeout=self.config['timeout'], flags=zmq.POLLIN):
                response = None
                try:
                    response = self.socket.recv_multipart()
                    if request == 'READY':
                        input_descr, input_data, output_descr = response
                        input_descr = [tuple(column) for column in json.loads(input_descr.decode())]
                        output_descr = [tuple(column[:2] + [tuple(column[2])]) for column
                                        in json.loads(output_descr.decode())]
                        self.input = np.frombuffer(input_data, dtype=input_descr)[0]
                        self.output = np.zeros(1, dtype=output_descr)[0]
                        self.logger.info('READY: received input data')
                        self.logger.debug(f'received: {np.frombuffer(input_data, dtype=input_descr)}')
                        return
                    else:
                        assert response[0] == b'ACK'
                        self.logger.debug(f'{request}: message acknowledged')
                        return
                except (ValueError, AssertionError):
                    self.logger.debug(f'{request}: received {response}')
                    self.logger.error(f'{request}: malformed reply')
            else:
                self.logger.warning(f'{request}: no response')
                tries += 1
                sleep(self.config['retry-sleep'])

            if tries >= self.config['retries'] + 1:
                self.logger.error(f'{request}: {tries} requests unsuccessful, abandoning')
                self.socket.close(0)
                raise ConnectionError('could not connect to RunnerInterface')

            # close and reopen the socket
            self.socket.close(linger=0)
            self.connect()
