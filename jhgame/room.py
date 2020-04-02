import uuid

from .jhserver import JankenHockeyServer
from .player import Player


class RoomFullError(Exception):
    pass


class RoomNotFoundError(Exception):
    pass


class NotInRoomError(Exception):
    pass


class ClientNotRegisteredError(Exception):
    pass


class Room:
    def __init__(self, id_, capacity, room_name, level="normal"):
        """Class for a room.

        Args:
            id_ (str): Room ID
            capacity (int): Room capacity
            room_name (str): Room name
            level (str, optional): Room level. Defaults to "normal".
        """
        self.id = id_
        self.capacity = capacity
        self.name = self.id if room_name is None else room_name
        self.players = []

        self.jhserver = JankenHockeyServer(level)

    def join(self, player):
        """Add player to room.

        Args:
            player (str): Player ID
        """
        if not self.is_full():
            self.players.append(player)
        else:
            raise RoomFullError()

    def leave(self, player):
        """Remove player from room.

        Args:
            player (str): Player ID
        """
        if player in self.players:
            self.players.remove(player)
        else:
            raise NotInRoomError()

    def is_empty(self):
        """Check if room is available."""
        return not bool(self.players)

    def is_full(self):
        """Check if the room is full."""
        return len(self.players) == self.capacity

    def is_in_room(self, player_id):
        """Check if the specified player is in the room.

        Args:
            player_id (str): Player ID
        """
        return player_id in [player.id for player in self.players]

    def get_player_ids(self):
        """Get a list of player IDs in the room."""
        return [player.id for player in self.players]


class Rooms:
    def __init__(self, capacity=2):
        """Control rooms.

        Args:
            capacity (int, optional): Room capacity. Defaults to 2.
        """
        self.room_capacity = capacity
        self.rooms = {}
        self.players = {}

    def register(self, address, udp_port):
        """Player registration.

        Args:
            address (tuple): Tuple containing address
            udp_port (int): UDP port

        Returns:
            str: Player ID
        """
        player = None
        for player in [player for registered_player in self.players.values() if registered_player.address == address]:
            player.udp_addr((address[0], udp_port))

        if player is None:
            player = Player(address, udp_port)
            self.players[player.id] = player

        return player

    def join(self, player_id, room_id=None):
        """Add player to room.

        Args:
            player_id (str): Player ID
            room_id (str, optional): Room ID. Defaults to None.

        Returns:
            str: Room ID
        """
        if player_id not in self.players:
            raise ClientNotRegisteredError()

        player = self.players[player_id]

        if room_id is None:
            room_id = self.create()

        if room_id in self.rooms:
            if not self.rooms[room_id].is_full():
                self.rooms[room_id].players.append(player)
                player.update_time()
                return room_id
            raise RoomFullError()
        raise RoomNotFoundError()

    def leave(self, player_id, room_id):
        """Remove player from room.

        Args:
            player_id (str): Player ID
            room_id (str): Room ID
        """
        if player_id not in self.players:
            raise ClientNotRegisteredError()

        player = self.players[player_id]

        if room_id in self.rooms:
            self.rooms[room_id].leave(player)
        else:
            raise RoomNotFoundError()

    def create(self, room_name=None, level="normal"):
        """Create a new room.

        Args:
            room_name (str, optional): Room name. Defaults to None.
            level (str, optional): Room level. Defaults to "normal".

        Returns:
            str: Room ID
        """
        id_ = str(uuid.uuid4())
        self.rooms[id_] = Room(id_, self.room_capacity, room_name, level)
        return id_

    def remove_empty(self):
        """Remove empty room."""
        empty_rooms = []
        for room_id in [room_id for room_id in self.rooms if self.rooms[room_id].is_empty()]:
            empty_rooms.append(room_id)
        for room_id in empty_rooms:
            self.rooms.pop(room_id)

    def send(self, player_id, room_id, message, socket):
        """Send message to client via UDP.

        Args:
            player_id (str): Player ID
            room_id (str): Room ID
            message (dict): Message to send
            socket (socket.Socket): Instance for Socket
        """
        if room_id not in self.rooms:
            raise RoomNotFoundError()

        room = self.rooms[room_id]
        if not room.is_in_room(player_id):
            raise NotInRoomError()

        for player in room.players:
            player.send_udp(player_id, message, socket)
