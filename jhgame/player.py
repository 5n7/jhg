import json
import time
import uuid


class Player:
    def __init__(self, address, udp_port, player_name=None):
        """Player information.

        Args:
            address (tuple): Tuple containing address
            udp_port (int): UDP port
            player_name (str, optional): Player name. Defaults to None.
        """
        self.address = address
        self.udp_addr = (address[0], int(udp_port))
        self.id = str(uuid.uuid4())
        self.name = self.id if player_name is None else player_name

        self.time = None

    @staticmethod
    def send_tcp(success, data, socket):
        """Send message to client via TCP."""
        success_string = "True" if success else "False"
        socket.send(json.dumps({"success": success_string, "message": data}).encode("UTF-8"))

    def send_udp(self, player_id, message, socket):
        """Send message to client via UDP."""
        socket.sendto(json.dumps({player_id: message}).encode("UTF-8"), self.udp_addr)

    def update_time(self):
        """Update the time of the last UDP communication."""
        self.time = time.time()

    def is_timeout(self):
        """Judge timeout."""
        # timeout if there is no time update for 3 seconds
        return time.time() - self.time > 3
