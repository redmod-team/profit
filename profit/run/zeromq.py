""" zeromq Interface

Ideas & Help from the 0MQ Guide (zguide.zeromq.org, examples are licensed with MIT)
"""

import zmq
import numpy as np
import json
from time import sleep
from logging import Logger
import os

from .interface import RunnerInterface, WorkerInterface


# === ZeroMQ Interface === #


class ZeroMQRunnerInterface(RunnerInterface, label="zeromq"):
    """Runner-Worker Interface using the lightweight message queue `ZeroMQ <https://zeromq.org/>`_

    - can use different transport systems, most commonly tcp
    - can be used efficiently on a cluster (tested)
    - expected to be inefficient for a large number of small, locally run simulations where communication overhead is
      a concern (unverified, could be mitigated by using a different transport system)
    - known issue: some workers were unable to establish a connection with three tries, reason unknown

    Parameters:
        transport: ZeroMQ transport protocol
        address: override ip address or hostname of the Runner Interface (default: localhost, automatic with Slurm)
        port: port of the Runner Interface
        connection: override for the ZeroMQ connection spec (Worker side)
        bind: override for the ZeroMQ bind spec (Runner side)
        timeout: connection timeout when waiting for an answer in seconds (Worker)
        retries: number of tries to establish a connection (Worker)
        retry_sleep: sleep time in seconds between each retry (Worker)

    Attributes:
        socket (zmq.Socket): ZeroMQ backend
        logger (logging.Logger): Logger
    """

    def __init__(
        self,
        size,
        input_config,
        output_config,
        *,
        transport="tcp",
        address=None,
        port=9000,
        connection=None,
        bind=None,
        timeout=4,
        retries=3,
        retry_sleep=1,
        logger_parent: Logger = None,
    ):
        if "FLAGS" not in [var[0] for var in self.internal_vars]:
            self.internal_vars += [("FLAGS", np.byte.__name__)]
        super().__init__(size, input_config, output_config, logger_parent=logger_parent)
        self.transport = transport
        self.address = address
        self.port = port
        self.connection = connection
        self._bind = bind
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

        self.socket = zmq.Context.instance().socket(zmq.ROUTER)
        self.socket.bind(self.bind)
        self.logger.info(f"connected to {self.bind}")

    @property
    def bind(self):
        if self._bind is None:
            return f"{self.transport}://*:{self.port}"
        else:
            return self._bind

    @bind.setter
    def bind(self, value):
        self._bind = value

    @property
    def config(self):
        config = {
            "transport": self.transport,
            "address": self.address,
            "port": self.port,
            "connection": self.connection,
            "bind": self._bind,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_sleep": self.retry_sleep,
        }
        return {**super().config, **config}  # super().config | config in python3.9

    def poll(self):
        self.logger.debug("polling: checking for messages")
        # poll does not wait for messages (timeout=10ms as 0 means wait forever)
        # waiting should be done with the runner (sleep)
        # this allows the runner to react to messages immediately
        while self.socket.poll(timeout=10, flags=zmq.POLLIN):
            msg = self.socket.recv_multipart()
            # ToDo: Heartbeats
            self.handle_msg(msg[0], msg[2:])

    def handle_msg(self, address: bytes, msg: list):
        if address[:4] == b"req_":  # req_123
            run_id = int(address[4:])
            if msg[0] == b"READY":
                input_descr = json.dumps(self.input_vars).encode()
                output_descr = json.dumps(self.output_vars).encode()
                self.logger.debug(
                    f"run {run_id} READY: {input_descr} + {self.input[run_id]} + output {output_descr}"
                )
                self.socket.send_multipart(
                    [address, b"", input_descr, self.input[run_id], output_descr]
                )
                self.internal["FLAGS"][run_id] |= 0x02
            elif msg[0] == b"DATA":
                self.output[run_id] = np.frombuffer(msg[1], dtype=self.output_vars)
                self.logger.debug(
                    f"run {run_id} DATA: {np.frombuffer(msg[1], dtype=self.output_vars)[0]}"
                )
                self.internal["DONE"][run_id] = True
                self.internal["FLAGS"][run_id] |= 0x08
                self.socket.send_multipart([address, b"", b"ACK"])  # acknowledge
            elif msg[0] == b"TIME":
                self.internal["TIME"][run_id] = np.frombuffer(msg[1], dtype=np.uint)
                self.logger.debug(
                    f"run {run_id} TIME: {np.frombuffer(msg[1], dtype=np.uint)[0]}"
                )
                self.socket.send_multipart([address, b"", b"ACK"])  # acknowledge
            elif msg[0] == b"DIE":
                self.internal["FLAGS"][run_id] |= 0x04
                self.logger.debug(f"run {run_id} DIE")
                self.socket.send_multipart([address, b"", b"ACK"])  # acknowledge
            else:
                self.logger.warning(f"received unknown message {address}: {msg}")
        else:
            self.logger.warning(
                f"received message from unknown client {address}: {msg}"
            )

    def clean(self):
        self.logger.info("cleaning: closing socket")
        self.socket.close(linger=0)
        zmq.Context.instance().destroy(linger=1)

    def __del__(self):
        self.socket.close(linger=0)
        zmq.Context.instance().destroy(linger=1)


class ZeroMQWorkerInterface(WorkerInterface, label="zeromq"):
    """Runner-Worker Interface using the lightweight message queue `ZeroMQ <https://zeromq.org/>`_

    counterpart to :py:class:`ZeroMQRunnerInterface`
    """

    def __init__(
        self,
        run_id: int,
        *,
        transport="tcp",
        address=None,
        port=9000,
        connection=None,
        bind=None,
        timeout=4,
        retries=3,
        retry_sleep=1,
        logger_parent: Logger = None,
    ):
        # TODO: duplicate default values
        super().__init__(run_id, logger_parent=logger_parent)
        self.transport = transport
        self.address = address
        self.port = port
        self._connection = connection
        self.bind = bind
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

        self._connected = False

    @property
    def connection(self):
        if self._connection is None:
            address = (
                self.address or os.environ.get("PROFIT_RUNNER_ADDRESS") or "localhost"
            )
            return f"{self.transport}://{address}:{self.port}"
        else:
            return self._connection

    @connection.setter
    def connection(self, value):
        self._connection = value

    @property
    def config(self):
        config = {
            "transport": self.transport,
            "address": self.address,
            "port": self.port,
            "connection": self._connection,
            "bind": self._bind,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_sleep": self.retry_sleep,
        }
        return {**super().config, **config}  # super().config | config in python3.9

    def retrieve(self):
        self.connect()
        self.request("READY")
        self.disconnect()

    def transmit(self):
        self.connect()
        self.request("TIME")
        self.request("DATA")
        self.disconnect()

    def clean(self):
        if self._connected:
            self.disconnect()

    def connect(self):
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, f"req_{self.run_id}".encode())
        self.socket.connect(self.connection)
        self.logger.info(f"connected to {self.connection}")
        self._connected = True

    def disconnect(self):
        self.socket.close(linger=0)
        self._connected = False

    def __del__(self):
        if self._connected:
            self.socket.close(linger=0)

    def request(self, request):
        """0MQ - Lazy Pirate Pattern"""
        if not self._connected:
            self.logger.info("no connection")
            self.connect()
        if request not in ["READY", "DATA", "TIME"]:
            raise ValueError(f'unknown request "{request}"')
        tries = 0
        while True:
            msg = [request.encode()]
            if request == "DATA":
                msg.append(self.output)
            elif request == "TIME":
                msg.append(np.uint(self.time))
            self.socket.send_multipart(msg)
            self.logger.debug(f"send message {msg}")
            if self.socket.poll(timeout=int(1e3 * self.timeout), flags=zmq.POLLIN):
                response = None
                try:
                    response = self.socket.recv_multipart()
                    if request == "READY":
                        input_descr, input_data, output_descr = response
                        input_descr = [
                            tuple(column) for column in json.loads(input_descr.decode())
                        ]
                        output_descr = [
                            tuple(column[:2] + [tuple(column[2])])
                            for column in json.loads(output_descr.decode())
                        ]
                        self.input = np.frombuffer(input_data, dtype=input_descr)[0]
                        self.output = np.zeros(1, dtype=output_descr)[0]
                        self.logger.info("READY - received input data")
                        self.logger.debug(
                            f"received: {np.frombuffer(input_data, dtype=input_descr)}"
                        )
                        return
                    else:
                        assert response[0] == b"ACK"
                        self.logger.info(f"{request}: message acknowledged")
                        return
                except (ValueError, AssertionError):
                    self.logger.debug(f"{request}: received {response}")
                    self.logger.error(f"{request}: malformed reply")
            else:
                self.logger.info(f"{request}: no response")
                tries += 1
                sleep(self.retry_sleep)

            if tries >= self.retries + 1:
                self.logger.error(
                    f"{request}: {tries} requests unsuccessful, abandoning"
                )
                self.disconnect()
                raise ConnectionError("could not connect to RunnerInterface")

            # close and reopen the socket
            self.disconnect()
            self.connect()
