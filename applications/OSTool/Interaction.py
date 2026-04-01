import asyncio
import glob
import json
import os
import re
import socket
import struct
import time
from typing import List, Dict, Any, Tuple

import docker
import docker.models.containers

class Container:
    def __init__(self, image):
        self.image = image
        self.client = docker.from_env()
        self.container: docker.models.containers.Container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
            labels={"created_by": "os-pipeline"},
        )
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash --login", stdin=True, tty=True
        )["Id"]
        self.sock = self.client.api.exec_start(self.exec_id, socket=True)._sock
        self.sock.settimeout(5)
        # clear buffer
        self.sock.recv(1000)

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass

    def execute(self, command: str):
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        # print("=== EXECUTING ===\n", command)
        if not isinstance(command, str):
            return DummyOutput(-1, b"")
        self.sock.send(command.encode("utf-8") + b"\n")
        # ignore input line
        data = self.sock.recv(8)
        _, n = struct.unpack(">BxxxL", data)
        _ = self.sock.recv(n)

        time_limit = 30  # seconds
        start_time = time.time()

        output = b""
        while True:
            if time.time() - start_time > time_limit:
                print(f"Time limit reached, breaking out of the loop. Command was: `{command}`")
                break
            try:
                data = self.sock.recv(8)
                # print(data)
                if not data:
                    break
                _, n = struct.unpack(">BxxxL", data)
                line = self.sock.recv(n)
                output += line
                if re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except TimeoutError:
                break
            except socket.timeout:
                break

        # Clean up the output by removing terminal control sequences, removes escape sequences starting with
        # ESC (0x1b), followed by...
        # ... any characters, an '@' character, any characters, ending with '#' or '$'
        output = re.sub(b"\x1b.+@.+[#|$] ", b'', output)
        # ... '[' and any combination of digits and semicolons, ending with a letter (a-z or A-Z)
        output = re.sub(b'\x1b\\[[0-9;]*[a-zA-Z]', b'', output)
        # ... ']' and any digits, a semicolon, any characters except BEL (0x07), and ending with BEL
        output = re.sub(b'\x1b\\][0-9]*;[^\x07]*\x07', b'', output)
        # ... '[?2004' and either 'h' or 'l'
        output = re.sub(b'\x1b\[\?2004[hl]', b'', output)

        # Remove BEL characters (0x07)
        output = re.sub(b'\x07', b'', output)

        return DummyOutput(0, output)

    def execute_independent(self, command, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python3", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(cmd)