import pymongo
import configparser

class MongoDB(object):
    def __init__(self):
        """
        A connection to our unstructured database.
        Configurations are set in the "settings.cfg" file (basic ConfigParser setup)

        """

        # Read the config
        self.cfg = configparser.ConfigParser()
        self.cfg.read("./settings.cfg")

        # Get the server location
        self.hostname = self.cfg.get('DB', 'host')
        self.port = int(self.cfg.get('DB', 'port'))

        # Get the auth info
        self.username = self.cfg.get('DB', 'username')
        self.password = self.cfg.get('DB', 'password')

        # Initiate the client
        self.client = pymongo.MongoClient("mongodb://%s:%s" % (self.hostname, self.port))
        self.databases = self.client.list_database_names()

# db = MongoDB()
