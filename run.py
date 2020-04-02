import argparse


def main(args):
    if args.is_server:
        from jhgame.deserver import DataExchangeServer

        deserver = DataExchangeServer(tcp_port=args.tcp_port, udp_port=args.udp_port)
        deserver.run()
    else:
        from jhgame.jhclient import JankenHockeyClient

        jhclient = JankenHockeyClient()
        jhclient.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janken Hockey")
    parser.add_argument(
        "-s", "--server", dest="is_server", action="store_true", help="Run server script",
    )
    parser.add_argument("-t", "--tcp-port", type=int, default=5000, help="TCP port")
    parser.add_argument("-u", "--udp-port", type=int, default=5000, help="UDP port")
    args = parser.parse_args()
    main(args)
