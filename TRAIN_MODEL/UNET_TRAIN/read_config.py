import configparser
import os

def read_config():
    cf=configparser.ConfigParser()
    cf.read('./config')
    return(cf.get)
