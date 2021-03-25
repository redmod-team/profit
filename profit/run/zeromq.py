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
import zmq.asyncio
import asyncio
import numpy as np
import json


@RunnerInterface.register('zeromq')
class ZeroMQRunnerInterface(RunnerInterface):
    def __init__(self, *args):
        if 'FLAGS' not in [var[0] for var in self.internal_vars]:
            self.internal_vars += [('FLAGS', np.bool8)]
        super().__init__(*args)
        self.socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
        self.socket.bind(self.config['bind'])
        self.logger.info('connected')
        self._loop = asyncio.create_task(self.loop())

    async def loop(self):
        self.logger.debug('start loop')
        while True:
            msg = await self.socket.recv_multipart()
            # ToDo: Heartbeats
            self.logger.debug(f'received message {msg[0]}: {msg[2:]}')
            await self.handle_msg(msg[0], msg[2:])

    async def handle_msg(self, address: bytes, msg: list):
        if address[:4] == b'req_':  # req_123
            run_id = int(address[4:])
            if msg[0] == b'READY':
                input_descr = json.dumps(self.input_vars).encode()
                output_descr = json.dumps(self.output_vars).encode()
                self.logger.debug(f'respond READY: input {input_descr} + {len(self.input.tobytes())} bytes, '
                                  f'output {output_descr}')
                await self.socket.send_multipart([address, b'', input_descr, self.input, output_descr])
                self.internal['FLAGS'][run_id] |= 0x02
            elif msg[0] == b'DATA':
                self.output[run_id] = np.frombuffer(msg[1], dtype=self.output_vars)
                self.internal['DONE'][run_id] = True
                self.internal['FLAGS'][run_id] |= 0x08
                self.logger.debug('acknowledge DATA')
                await self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'TIME':
                self.internal['TIME'][run_id] = msg[1]
                self.logger.debug('acknowledge TIME')
                await self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            elif msg[0] == b'DIE':
                self.internal['FLAGS'][run_id] |= 0x04
                self.logger.debug('acknowledge DIE')
                await self.socket.send_multipart([address, b'', b'ACK'])  # acknowledge
            else:
                self.logger.warning(f'received unknown message {address}: {msg}')
        else:
            self.logger.warning(f'received message from unknown client {address}: {msg}')

    def cancel(self, run_id): # Todo: redo
        if not self.internal['FLAGS'][run_id] & 0b11100:
            if self.internal['FLAGS'][run_id] & 0b10:
                self.socket.send_multipart([b'rep' + bytes([run_id]), b'', b'CANCEL'])
                self.internal['FLAGS'][run_id] |= 0x10  # cancelled
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
        self.ready = asyncio.create_task(self.request('READY'))  # self.input, self.output

    @property
    def time(self):
        return self._time

    @time.setter
    async def time(self, value: int):
        self._time = value
        await self.request('TIME')

    async def done(self):
        await self.request('DATA')

    def connect(self, who=None):
        if who is None:
            self.connect('req')
            self.connect('rep')
            return self.req, self.rep
        elif who == 'req':
            self.req = zmq.asyncio.Context.instance().socket(zmq.REQ)
            self.req.setsockopt(zmq.IDENTITY, f'req_{self.worker.run_id}'.encode())
            self.req.connect(self.config['connect'])
            self.logger.info('REQ connected')
        elif who == 'rep':
            self.rep = zmq.asyncio.Context.instance().socket(zmq.REP)
            self.rep.setsockopt(zmq.IDENTITY, f'rep_{self.worker.run_id}'.encode())
            self.req.connect(self.config['connect'])
            self.logger.info('REP connected')
        else:
            raise ValueError

    async def request(self, request):
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
            self.req.send_multipart(msg)
            if await self.req.poll(self.REQUEST_TIMEOUT) & zmq.POLLIN:
                try:
                    response = await self.req.recv_multipart()
                    self.logger.debug(f'REQ {request}: received {response}')
                    if request == 'READY':
                        input_descr, input_data, output_descr = response
                        input_descr = [tuple(column) for column in json.loads(input_descr.decode())]
                        output_descr = [tuple(column[:2] + [tuple(column[2])]) for column
                                        in json.loads(output_descr.decode())]
                        self.input = np.frombuffer(input_data, dtype=input_descr)[0]
                        self.output = np.zeros(1, dtype=output_descr)[0]
                        self.logger.info('REQ READY: received input data')
                        return True
                    else:
                        assert response[0] == b'ACK'
                        self.logger.info(f'REQ {request}: message acknowledged')
                        return True
                except (ValueError, AssertionError):
                    self.logger.error(f'REQ {request}: malformed reply')
            else:
                self.logger.warning(f'REQ {request}: no response')
                tries += 1
                await asyncio.sleep(1)

            if tries >= self.REQUEST_TRIES:
                self.logger.error(f'REQ {request}: {tries} requests unsuccessful, abandoning')
                self.worker.cancel()
                return False

            # close and reopen the socket
            self.req.close(linger=0)
            self.connect('req')
