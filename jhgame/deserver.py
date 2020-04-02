import json
import socket
import threading
import time

from .room import NotInRoomError, RoomFullError, RoomNotFoundError, Rooms

BUFFERSIZE = 1024


class UDPServerError(Exception):
    pass


class _TCPServer(threading.Thread):
    def __init__(self, tcp_port, rooms, lock):
        """TCP server.

        Args:
            tcp_port (int): TCP Port
            rooms (Rooms): Instance for Room
            lock (threading.Lock): Instance for Lock
        """
        threading.Thread.__init__(self)

        self._tcp_port = tcp_port
        self._rooms = rooms
        self._lock = lock

        self._socket = None
        self._message = '{"success": "{success}", "message": "{message}"}'

        self.is_listening = True

    def run(self):
        """Run TCP server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("0.0.0.0", self._tcp_port))
        self._socket.settimeout(5)
        self._socket.listen(1)
        time_ref = time.time()

        while self.is_listening:
            if time_ref + 1 < time.time():
                self._lock.acquire()
                try:
                    # judge timeout for all players
                    for room_id, room in self._rooms.rooms.items():
                        for player_id, player in [
                            (player_id, self._rooms.players[player_id]) for player_id in room.get_player_ids()
                        ]:
                            if player.is_timeout():
                                self._rooms.leave(player_id, room_id)
                    self._rooms.remove_empty()
                finally:
                    self._lock.release()
                time_ref = time.time()
            try:
                conn, address = self._socket.accept()
                data = conn.recv(BUFFERSIZE)
            except socket.timeout:
                continue

            try:
                data = json.loads(data.decode("UTF-8"))
            except ValueError:
                print(f"Error: The received message from {address} is broken.")

            try:
                action = data["action"]
                id_ = data.get("id")
                room_id = data.get("room_id")
                payload = data.get("payload")

                self._lock.acquire()
                try:
                    print(data)
                    self.route(conn, address, action, payload, id_, room_id)
                finally:
                    self._lock.release()
            except KeyError:
                print(f"Error: The received message from {address} is broken.")

            conn.close()

        self.stop()

    def route(self, sock, address, action, payload, id_=None, room_id=None):
        """Routing data received over TCP.

        Args:
            sock (socket.Socket): Instance for Socket
            address (tuple): Tuple containing address
            action (str): TCP action
            payload (int/dict): Payload
            id_ (str, optional): Client ID. Defaults to None.
            room_id (str, optional): Room ID. Defaults to None.
        """
        if action == "register":
            client = self._rooms.register(address, payload)
            client.send_tcp(True, client.id, sock)
            return

        if id_ is not None:
            if id_ not in self._rooms.players.keys():
                print(f"Error: {id_} is not registered " + f"({address[0]}:{address[1]}).")
                sock.send(self._message.format(success="False", message="unknown_id"))
                return

            client = self._rooms.players[id_]

            if action == "join":
                try:
                    if payload not in self._rooms.rooms.keys():
                        raise RoomNotFoundError()
                    self._rooms.join(id_, payload)
                    client.send_tcp(True, payload, sock)
                except RoomNotFoundError:
                    client.send_tcp(False, room_id, sock)
                except RoomFullError:
                    client.send_tcp(False, room_id, sock)

            elif action == "get_rooms":
                rooms = []
                for id_room, room in self._rooms.rooms.items():
                    rooms.append(
                        {
                            "id": id_room,
                            "name": room.name,
                            "capacity": room.capacity,
                            "num_players": len(room.players),
                            "level": room.jhserver.level,
                        }
                    )
                client.send_tcp(True, rooms, sock)

            elif action == "create":
                room_id = self._rooms.create(payload["room_name"], payload["level"])
                self._rooms.join(client.id, room_id)
                client.send_tcp(True, room_id, sock)

            elif action == "leave":
                try:
                    if room_id not in self._rooms.rooms:
                        raise RoomNotFoundError()
                    self._rooms.leave(id_, room_id)
                    client.send_tcp(True, room_id, sock)
                except RoomNotFoundError:
                    client.send_tcp(False, room_id, sock)
                except NotInRoomError:
                    client.send_tcp(False, room_id, sock)

            else:
                sock.send_tcp(self._message.format(success="False", message="not_registered"))

    def stop(self):
        """Stop TCP server."""
        self._socket.close()


class _UDPServer(threading.Thread):
    def __init__(self, udp_port, rooms, lock):
        """UDP Server.

        Args:
            udp_port (int): UDP port
            rooms (Rooms): Instance fot Rooms
            lock (threading.Lock): Instance fot Lock
        """
        threading.Thread.__init__(self)

        self._udp_port = udp_port
        self._rooms = rooms
        self._lock = lock

        self._socket = None
        self._message = '{"success": "{success}", "message": "{message}"}'

        self.is_listening = True

    def run(self):
        """Run UDP server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("0.0.0.0", self._udp_port))
        self._socket.settimeout(5)

        while self.is_listening:
            try:
                data, address = self._socket.recvfrom(BUFFERSIZE)
            except socket.timeout:
                continue

            try:
                data = json.loads(data.decode("UTF-8"))
            except ValueError:
                print(f"Error: The received message from {address} is broken.")

            try:
                action = data.get("action")
                id_ = data.get("id")
                room_id = data.get("room_id")
                payload = data.get("payload")

                try:
                    if room_id not in self._rooms.rooms.keys():
                        raise RoomNotFoundError
                    room = self._rooms.rooms[room_id]
                    if not room.is_in_room(id_):
                        raise NotInRoomError
                    self._lock.acquire()

                    try:
                        if action == "send":
                            try:
                                send_msg = room.jhserver.call(
                                    id_, payload["message"], room.get_player_ids(),
                                )  # dictionary to send to both clients
                                self._rooms.send(id_, room_id, send_msg, self._socket)
                                # update the time of the last UDP communication
                                self._rooms.players[id_].update_time()

                                # judge timeout for all players in each room
                                for room_id, room in self._rooms.rooms.items():
                                    for player_id, player in [
                                        (player_id, self._rooms.players[player_id],)
                                        for player_id in room.get_player_ids()
                                    ]:
                                        if player.is_timeout():
                                            self._rooms.leave(player_id, room_id)
                            except Exception:
                                raise UDPServerError
                    finally:
                        self._lock.release()
                except RoomNotFoundError:
                    raise RoomNotFoundError("Room not found.")
                except NotInRoomError:
                    raise NotInRoomError("Player is not in room.")

            except KeyError:
                print(f"Error: The received .json from {address} is broken.")

        self.stop()

    def stop(self):
        """Stop UDP server."""
        self._socket.close()


class DataExchangeServer:
    def __init__(self, tcp_port=5000, udp_port=5000):
        """Janken Hockey Game Data Exchange Server.

        Args:
            tcp_port (int, optional): TCP port. Defaults to 5000.
            udp_port (int, optional): UDP port. Defaults to 5000.
        """
        self._lock = threading.Lock()
        self._rooms = Rooms()
        self._tcp_server = _TCPServer(tcp_port, self._rooms, self._lock)
        self._udp_server = _UDPServer(udp_port, self._rooms, self._lock)

    def run(self):
        """Run game server."""
        self._tcp_server.start()
        self._udp_server.start()

        print(
            """Janken Hockey Game Server
----------------------------------------
Usage:
    list: Show room list
    room [room_id]: Show selected room's information
    player [player_id]: Show selected player's information
    quit: Stop game server
        """
        )

        is_running = True
        try:
            while is_running:
                cmd = input("JHS > ")
                if cmd == "list":
                    print("Room list:")
                    try:
                        for room in self._rooms.rooms.values():
                            print(f"{room.id} - {room.name} " + f"({len(room.players)}/{room.capacity})")
                    except Exception:
                        print("Error: Failed to get room list.")

                elif cmd.startswith("room "):
                    print("Room information:")
                    try:
                        id_ = cmd[5:]
                        room = self._rooms.rooms.get(id_)
                        if room is not None:
                            print(f"{room.id} - {room.name} " + f"({len(room.players)}/{room.capacity})")

                            print("Players:")
                            for player in room.players:
                                print(player.id)
                        else:
                            print(f"Room ID {id_} not found.")
                    except Exception:
                        print("Error: Failed to get room information.")

                elif cmd.startswith("player "):
                    print("Player information:")
                    try:
                        idx = cmd[7:]
                        player = self._rooms.players[int(idx)]
                        print(f"{player.id} : " + f"{player.udp_addr[0]}:{player.udp_addr[1]}")
                    except Exception:
                        print("Error: Failed to get player information.")

                elif cmd in ("quit", "q"):
                    print("Shutting down server...")
                    self._udp_server.is_listening = False
                    self._tcp_server.is_listening = False
                    is_running = False

        except KeyboardInterrupt:
            print("Shutting down server...")
            self._udp_server.is_listening = False
            self._tcp_server.is_listening = False

        finally:
            self._udp_server.join()
            self._tcp_server.join()
