import hashlib

def md5sum(file_url):

    md5sum = hashlib.md5()
    with open(file_url,'rb') as f:

        while True:

            data = f.read(2048)
            if not data:
                break

            md5sum.update(data)
    
    return md5sum.hexdigest()