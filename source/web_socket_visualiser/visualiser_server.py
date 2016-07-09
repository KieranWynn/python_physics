import collections

import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
'''
This is a simple Websocket Echo server that uses the Tornado websocket handler.
Please run `pip install tornado` with python of version 2.7.9 or greater to install tornado.
This program will echo back the reverse of whatever it recieves.
Messages are output to the terminal for debugging purposes.
'''


class MessageQueue(collections.deque):
    """
    A FILO queue with the front of the queue at the right.
    New elements are added at the left.
    """
    def fetch(self, n=None):
        """
        Generator to remove and return the first n elements in the queue
        @param n: number of elements to fetch
        @yield: each element as it is removed
        """
        if not n:
            n = len(self)
        for i in range(n):
            yield self.pop()

    def add(self, iterable):
        self.extendleft(iterable)

class WSHandler(tornado.websocket.WebSocketHandler):

    inbox = MessageQueue() # incoming message queue. The front of the queue is on the right, new messages are added on the left
    outbox = MessageQueue() # outgoing message queue. The front of the queue is on the right, new messages are added on the left
    clients = []

    def open(self):
        print('New client connected!')
        self.write_message("Welcome, new client")
        self.clients.append(self) # appends to class variable list of clients

    def on_message(self, message):
        print('message received:  %s' % message)
        self.inbox.extendleft([message])

    def on_close(self):
        self.clients.remove(self)
        print('Client disconnected')

    def check_origin(self, origin):
        return True

    @classmethod
    def write_to_clients(cls):
        for message in cls.outbox.fetch():
            for client in cls.clients:
                client.write_message(message)

application = tornado.web.Application([
    (r'/ws', WSHandler),
])

class VisualiserServer(object):

    def __init__(self, update_callback, update_interval_ms=100, port=8888):
        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(port) #default 8888
        myIP = socket.gethostbyname(socket.gethostname())
        print('*** Websocket Server Started at %s***' % myIP)

        self.main_loop = tornado.ioloop.IOLoop.instance()
        self.schedule = tornado.ioloop.PeriodicCallback(update_callback, update_interval_ms, io_loop=self.main_loop)

    def fetch_messages(self):
        return WSHandler.inbox.fetch()

    def send_messages(self, messages):
        WSHandler.outbox.add(messages)
        WSHandler.write_to_clients()

    def start(self):
        self.schedule.start()
        self.main_loop.start()
