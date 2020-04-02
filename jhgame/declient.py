import json
import socket
import threading

BUFFERSIZE = 1024


class DataExchangeClientError(Exception):
    pass


class _UDPClient(threading.Thread):
    def __init__(self, declient):
        """UDP client for data reception.

        Args:
            declient (DataExchangeClient): Instance for DataExchangeClient
        """
        threading.Thread.__init__(self)

        self._declient = declient
        self._lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(declient.client_udp)
        self.socket.settimeout(5)

        self.is_listening = True

    def run(self):
        """Receive data from server."""
        while self.is_listening:
            try:
                data, address = self.socket.recvfrom(BUFFERSIZE)
            except socket.timeout:
                continue

            try:
                data = json.loads(data.decode("UTF-8"))
            except ValueError:
                print(f"Error: The received message from {address} is broken.")

            self._lock.acquire()
            try:
                self._declient.message.append(data)
            finally:
                self._lock.release()

        self._stop()

    def _stop(self):
        """Stop UDP client."""
        self.socket.close()


class DataExchangeClient:
    def __init__(
        self, server_host="127.0.0.1", server_port_tcp=5000, server_port_udp=5000, client_port_udp=5001,
    ):
        """Client for sending and receiving data.

        Args:
            server_host (str, optional): Server host IP address.
                Defaults to "127.0.0.1".
            server_port_tcp (int, optional): Server TCP port. Defaults to 5000.
            server_port_udp (int, optional): Server UDP port. Defaults to 5000.
            client_port_udp (int, optional): Client UDP port. Defaults to 5001.
        """
        self.id = None
        self.room_id = None
        self.message = []

        self.client_udp = ("0.0.0.0", client_port_udp)
        self.server_tcp = (server_host, server_port_tcp)
        self.server_udp = (server_host, server_port_udp)
        self._udp_client = _UDPClient(declient=self)

        self._udp_client.start()
        self.register()

    @staticmethod
    def _parse_data(data):
        """Get message from data received from server."""
        try:
            data = json.loads(data.decode("UTF-8"))
        except ValueError:
            print(f"Error: The received message is broken.")

        if data["success"] == "True":
            return data["message"]
        raise Exception(data["message"])

    @staticmethod
    def _send_message_tcp(message, server_tcp):
        """Send and receive messages to the server via TCP."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(server_tcp)
            sock.send(message)
            data = sock.recv(BUFFERSIZE)
        return data

    def register(self):
        """Register the client on the server."""
        message = json.dumps({"action": "register", "payload": self.client_udp[1]}).encode("UTF-8")
        self.id = self._parse_data(self._send_message_tcp(message, self.server_tcp))
        return self.id

    def create_room(self, room_name, level):
        """Create a new room on the server.

        Args:
            room_name (str): Room name
            level (str): Room level
        """
        if level in ("easy", "normal", "hard", "god"):
            message = json.dumps(
                {"action": "create", "payload": {"room_name": room_name, "level": level}, "id": self.id,}
            ).encode("UTF-8")
            self.room_id = self._parse_data(self._send_message_tcp(message, self.server_tcp))
        else:
            print("Room level must be 'easy', 'normal', 'hard', or 'god'.")

    def join_room(self, room_id):
        """Join specified room on server.

        Args:
            room_id (str): Room ID

        Returns:
            bool: Whether the room was successfully joined
        """
        message = json.dumps({"action": "join", "payload": room_id, "id": self.id}).encode("UTF-8")
        try:
            self.room_id = self._parse_data(self._send_message_tcp(message, self.server_tcp))
        except DataExchangeClientError:
            return False
        return True

    def leave_room(self):
        """Leave current room."""
        message = json.dumps({"action": "leave", "room_id": self.room_id, "id": self.id}).encode("UTF-8")
        self._send_message_tcp(message, self.server_tcp)

    def get_rooms(self):
        """Get room list on server."""
        message = json.dumps({"action": "get_rooms", "id": self.id}).encode("UTF-8")
        return self._parse_data(self._send_message_tcp(message, self.server_tcp))

    def send(self, message):
        """Send message to server via UDP."""
        message = json.dumps(
            {"action": "send", "payload": {"message": message}, "room_id": self.room_id, "id": self.id,}
        ).encode("UTF-8")
        self._udp_client.socket.sendto(message, self.server_udp)

    def get_messages(self):
        """Get message received from server."""
        message = self.message
        self.message = []  # delete accumulated messages
        return message
