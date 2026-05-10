import os

def validate_file_extension(filename, extensions=('.txt', '.csv')):
    return filename.lower().endswith(extensions)