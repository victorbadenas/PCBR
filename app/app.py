from flask import Flask, request

app = Flask(__name__)

class PCBR:
    def __init__(self):
        pass

    def test(self, *args, **kwargs):
        return 'test'

pcbr = PCBR()

@app.route("/test")
def test():
    return pcbr.test()

if __name__ == "__main__":
    app.run()
